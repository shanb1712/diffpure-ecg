# Diffusion model for adversarial attacks on ECG

As part of a private project, I created an adversarial model on timer series (ECG) and evaluated it against bpda+eot attack. This is an initial attemp to create and run adversarial attacks on ECG defense model and can be further extended to other attacks.

## Data
### ECG signals
TNMG (Telehealth Network of Minas Gerais)  database from Brazil was used to train and evaluate this framework. It consist over 2M 12-lead ECG recordings of 10 sec each with 6 different arrhythmias (1dAVb, RBBB, LBBB, SB, AF, ST). For more information refer to: [Automatic diagnosis of the 12-lead ECG using a deep neural network](https://www.nature.com/articles/s41467-020-15432-4) and [Improving patient access to specialized health care: the Telehealth Network of Minas Gerais, Brazil](https://www.scielosp.org/article/bwho/2012.v90n5/373-378/).

- **Evaluating**: To evaluate the pre-trained models, the *testing dataset* must be downloaded. 
- **Training**: To re-trained the model the *full CODE dataset* should also be downloaded. 

Follow section Datasets on [automatic-ecg-diagnosis](https://github.com/antonior92/automatic-ecg-diagnosis).

### Baseline wander and noise
The [MIT-BIH Noise Stress Test Database](https://physionet.org/content/nstdb/1.0.0/) was used to corrupt the ECG signals in the training the diffusion model. To download the model run
 
```
bash ./data/download_data.sh
```

## Models
### Classifier
I used the ResNet model developed in [automatic-ecg-diagnosis](https://github.com/antonior92/automatic-ecg-diagnosis). The model was modified to classify six different arrhythmias from one lead instead of 12-leads. It can be trained using the script `run_classifier.py`. Alternatively, pre-trained model is saved under `check_points/resnet_model`.

### Diffusion model 
As for diffusion model I used the ddpm implementation (DeScoD-ECG) for ECG wander and noise removal from [Score-based-ECG-Denoising](https://github.com/HuayuLiArizona/Score-based-ECG-Denoising/tree/main). 
The model was trained on 1000 examples from the TNMG database due to time restrictions, however it can definitely be trained on all database. The signals were further corrupted with baseline wander noise profiles described above.
Two models were trained based on two different noise type (1 or 2
), and the best was selected for the final framework.
To train the model run the `run_diffusion.py`. Otherwise, pre-trained diffusion models are saved under `check_points/noise_type_{1/2}`.

## Attacks
The BPDA+EOT was applied as described in [Stochastic security: adversarial defense using long-run dynamics on energy-based models](https://arxiv.org/abs/2005.13525).

*Further attacks can be implemented and explored*.

## Scripts
- Train: to run from scratch both classifier and diffusion model should be trained. The following scripts can be run in parallel:
```
python3 run_classifier.py --path_to_database PATH_TO_TNMG_DATA --train
python3 run_diffusion.py --path_to_database PATH_TO_TNMG_DATA --n_type 1 --train
```
Choose `--n_type` as 1 or 2.

- Evaluate: To evaluate models, the following should be run:
```
python3 run_classifier.py --path_to_database PATH_TO_TNMG_DATA
python3 run_diffusion.py --path_to_database PATH_TO_TNMG_DATA --n_type 1/2
```
Final models will be then saved under `check_points`.

- Run experiment

To run the BDPA+EOT attack on the final model:
```
python3 eval_sde_adv_bpda.py --config tnmg.yml
```

## Results
