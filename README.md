# CVD_swin
our paper " Image Recoloring for Color Vision Deficiency Compensation Using Swin Transformer".
CVD, recoloring 


## How to train

### install the requirement using conda vitural enviroment
conda env create -f 190pytorch_environment.yml

### download the dataset
https://drive.google.com/drive/folders/10WMXPbtpV7Hy5_qBA_TCEbW-kCpj1D7v?usp=sharing

put the dataset in the same directory. 
### train
python train.py --epoch 0 --n_epochs 121 --batch_size 8 --checkpoint_interval 10 --dataset_name cvd_100_001 --sample 5000  --ssimori 1 --cvd 1 --lambda_ssim 0.00 --points 3000

the model will be saved in ./saved_models
the images will be saved in ./images

### result in paper.
![image](https://github.com/Ligeng-c/CVD_swin/blob/main/readme_imgs/P_image_1.png)
![image](https://github.com/Ligeng-c/CVD_swin/blob/main/readme_imgs/D_image_1.png)

