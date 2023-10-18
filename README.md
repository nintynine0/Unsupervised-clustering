# Unsupervised-clustering

This a readme file for detection and unsupervised clustering scripts.

Written at 11:29, 6 Sep. 2023 (Beijing Time), by Yanlan Hu (USTC).

Email: yanlan@mail.ustc.edu.cn

## Instructions:

### 1. Running environment

   Ubuntu_20.04 + cuda_11.2

   Pytorch install:
   ```
   pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
   ```
   
   Other necessary python packages are listed in [requirements.txt]. You can install them manually or run 'pip install -r requirements.txt'.

### 2. The algorithm flow
```
 continuous waveform
           |
           |
           |
           v
    [build_catalog.py]: detect events by STALTA and transform them into 4s spectrograms
                        save them in main_path/events
           |
           |
           |
           v
    [AutoEncoder.py]: use the spectrograms as input of Autoencoder and get the latent feature of each sample
                      save by main_path/out/feature_loss.dat
           |
           |
           |
           v
       [GMM.py]: apply gaussian mixture model to cluster the features
                 clustering results are saved in main_path/out and main_path/out/examples
```




### 3. U need to modify

(1) all the absolute paths in all files

(2) [config.py]

station_list if there are more than 1 station

the t array and f array of your spectrograms (copy them from the print of build_catalog.py)

(3) [AutoEncoder.py]

shape of spectrogram /Line 200/ (copy from the print of build_catalog.py)
