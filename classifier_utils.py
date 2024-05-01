import pandas as pd
import numpy as np
import torch
from torchmetrics import AveragePrecision, Recall, Specificity, F1Score, PrecisionRecallCurve

from tqdm import tqdm

score_fun = {
    'Recall': Recall(task="binary"),
    'Specificity': Specificity(task="binary"),
    'F1 score': F1Score(task="binary")}


def get_data(y, traces_ids):
    # ----- Data settings ----- #
    diagnosis = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]
    # ------------------------- #
    y.set_index('id_exam', drop=True, inplace=True)
    y = y.reindex(traces_ids, copy=False)
    df_diagnosis = y.reindex(columns=[d for d in diagnosis])
    y = df_diagnosis.values
    return y


# %% Auxiliar functions
def get_scores(y_true, y_pred, score_fun):
    nclasses = np.shape(y_true)[1]
    scores = []
    for name, fun in score_fun.items():
        scores += [[fun(y_pred[:, k], y_true[:, k]) for k in range(nclasses)]]
    return np.array(scores).T


def get_optimal_precision_recall(y_true, y_score):
    """Find precision and recall values that maximize f1 score."""
    n = np.shape(y_true)[1]
    opt_precision = []
    opt_recall = []
    opt_threshold = []
    pr_curve = PrecisionRecallCurve(task="binary")
    for k in range(n):
        # Get precision-recall curve
        precision, recall, threshold = pr_curve(y_score[:, k], y_true[:, k])
        # Compute f1 score for each point (use nan_to_num to avoid nans messing up the results)
        f1_score = np.nan_to_num(2 * precision * recall / (precision + recall))
        # Select threshold that maximize f1 score
        index = np.argmax(f1_score)
        opt_precision.append(precision[index])
        opt_recall.append(recall[index])
        t = threshold[index - 1] if index != 0 else threshold[0] - 1e-10
        opt_threshold.append(t)
    return np.array(opt_precision), np.array(opt_recall), np.array(opt_threshold)


def load_annotators(path_to_files):
    # Get true values
    y_true = pd.read_csv(path_to_files / 'gold_standard.csv').values
    # Get two annotators
    y_cardiologist1 = pd.read_csv(path_to_files / 'cardiologist1.csv').values
    y_cardiologist2 = pd.read_csv(path_to_files / 'cardiologist2.csv').values
    # Get residents and students performance
    y_cardio = pd.read_csv(path_to_files / 'cardiology_residents.csv').values
    y_emerg = pd.read_csv(path_to_files / 'emergency_residents.csv').values
    y_student = pd.read_csv(path_to_files / 'medical_students.csv').values
    return y_true, y_cardiologist1, y_cardiologist2, y_cardio, y_emerg, y_student


def report_performance(output_file, path_to_annotators, y_pred):
    y_true, y_cardiologist1, y_cardiologist2, y_cardio, y_emerg, y_student = load_annotators(path_to_annotators)
    _, _, our_threshold = get_optimal_precision_recall(torch.from_numpy(y_true), torch.from_numpy(y_pred))
    mask = y_pred > our_threshold
    y_ours = np.zeros_like(y_pred, dtype=int)
    y_ours[mask] = 1

    # evaluation metrics
    diagnosis = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']
    nclasses = len(diagnosis)
    predictor_names = ['DNN', 'cardio.', 'emerg.', 'stud.']

    # Get micro average precision
    average_precision = AveragePrecision(task="multilabel", num_labels=nclasses, average='micro')
    micro_avg_precision_ours = average_precision(torch.from_numpy(y_pred[:, :6]), torch.from_numpy(y_true[:, :6]))

    print('\nMicro average precision Ours:')
    print(micro_avg_precision_ours)
    # %% Generate table with scores for the average model (Table 2)
    scores_list = []
    for y in [y_ours, y_cardio, y_emerg, y_student]:
        # Compute scores
        scores = get_scores(torch.from_numpy(y_true), torch.from_numpy(y), score_fun)
        # Put them into a data frame
        scores_df = pd.DataFrame(scores, index=diagnosis, columns=score_fun.keys())
        # Append
        scores_list.append(scores_df)
    # Concatenate dataframes
    scores_all_df = pd.concat(scores_list, axis=1, keys=predictor_names)

    # Change multiindex levels
    scores_all_df = scores_all_df.swaplevel(0, 1, axis=1)
    scores_all_df = scores_all_df.reindex(level=0, columns=score_fun.keys())

    # Save results
    scores_all_df.to_excel(f"{str(output_file)}.xlsx", float_format='%.3f')
    scores_all_df.to_csv(f"{str(output_file)}.csv", float_format='%.3f')
    return


def train(model, ep, train_loader, device, criterion, optimizer):
    model.train()
    total_loss = 0
    n_entries = 0
    train_desc = "Epoch {:2d}: train - Loss: {:.6f}"
    train_bar = tqdm(initial=0, leave=True, total=len(train_loader),
                     desc=train_desc.format(ep, 0, 0), position=0)
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.reshape(batch_x.shape[0], 1, batch_x.shape[1])
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        # Reinitialize grad
        model.zero_grad()
        # Send to device
        # Forward pass
        pred_y = model(batch_x)
        loss = criterion(pred_y, batch_y.float())
        # Backward pass
        loss.backward()
        # Optimize
        optimizer.step()
        # Update
        bs = len(batch_x)
        total_loss += loss.detach().cpu().numpy()
        n_entries += bs
        # Update train bar
        train_bar.desc = train_desc.format(ep, total_loss / n_entries)
        train_bar.update(1)
    train_bar.close()
    return total_loss / n_entries


def evaluate(model, ep, test_loader, device, criterion):
    model.eval()
    total_loss = 0
    n_entries = 0
    eval_desc = "Epoch {:2d}: valid - Loss: {:.6f}"
    eval_bar = tqdm(initial=0, leave=True, total=len(test_loader),
                    desc=eval_desc.format(ep, 0, 0), position=0)
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.reshape(batch_x.shape[0], 1, batch_x.shape[1])
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        with torch.no_grad():
            # Forward pass
            pred_y = model(batch_x)
            loss = criterion(pred_y, batch_y.float())
            # Update outputs
            bs = len(batch_x)
            # Update ids
            total_loss += loss.detach().cpu().numpy()
            n_entries += bs
            # Print result
            eval_bar.desc = eval_desc.format(ep, total_loss / n_entries)
            eval_bar.update(1)
    eval_bar.close()
    return total_loss / n_entries
