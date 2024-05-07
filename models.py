import torch
from resnetEcg import ResNet1d
import os


def load_resnetEcg(config, path):
    # Get checkpoint
    if os.path.exists(path):
        ckpt = torch.load(f"{path}/model.pth", map_location=lambda storage, loc: storage)
    else:
        print(f'Could not find pretrained model in: {path}.')

    # Get model
    N_LEADS = config.n_leads
    N_CLASSES = 6
    model = ResNet1d(input_dim=(N_LEADS, config.seq_length),
                     blocks_dim=list(zip(config.net_filter_size, config.net_seq_lengh)),
                     n_classes=N_CLASSES,
                     kernel_size=config.kernel_size,
                     dropout_rate=config.dropout_rate)
    # load model checkpoint
    model.load_state_dict(ckpt["model"])
    return model
