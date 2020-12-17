# Correction-Filter

The official implementation of the work "Correction Filter for Single Image Super-Resolution: Robustifying Off-the-Shelf Deep Super-Resolvers" (https://arxiv.org/abs/1912.00157 , Accepted to CVPR 2020 - oral)

# Non-Blind
1. Downsample images and put into folders according to which down-sampling filter was used (it is recommended to use '.mat' for saving the LR images).
2. Define the sampling and reconstruction basis (s and r) in "correct_imgs.py" (lines 25-33).
3. Run: python correct_imgs.py --in_dir "Directory to the folder of the LR images" --out_dir "Directory of where to save the corrected images" --scale_factor "SR scale factor".
4. Run any off-the-shelf deep SR network trained using r (usually bicubic) on the images saved to out_dir

Note that this code assumes that the images within a folder are sampled using the same kernel.

# Blind
1. Define the SR network in "estimate_correction.py".
2. Run: estimate_correction.py --scale_factor "SR scale factor" --in_dir "Directory of the LR images" --out_dir "Directory to save the LR corrected images in it"
4. Run any off-the-shelf deep SR network trained using r (usually bicubic) on the images saved to out_dir

# Citation:

    @ARTICLE{correction_filter,
      author = {{Abu Hussein}, Shady and {Tirer}, Tom and {Giryes}, Raja},
      title = "{Correction Filter for Single Image Super-Resolution: Robustifying Off-the-Shelf Deep Super-Resolvers}",
      journal = {In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      year = "2020"
    }

# Results
## Non-Blind Super-Resolution
Non-blind super-resolution with scale factor of 4 on Gaussian model with std 4.5/sqrt(2) (left is DBPN without correction, right is with correction filter)

<img width="400" src="https://github.com/shadyabh/Correction-Filter/blob/master/figs/SR/baboon_Gauss_std3.2_x4_s.png"> <img width="400" src="https://github.com/shadyabh/Correction-Filter/blob/master/figs/SR/baboon_Gauss_std3.2_x4_s_x4_corr_corrected.png">

<img width="400" src="https://github.com/shadyabh/Correction-Filter/blob/master/figs/SR/zebra_Gauss_std3.2_x4_s.png"> <img width="400" src="https://github.com/shadyabh/Correction-Filter/blob/master/figs/SR/zebra_Gauss_std3.2_x4_s_x4_corr_corrected.png">

Non-blind super-resolution with scale factor of 2 on Gaussian model with std 2.5/sqrt(2) (left is DBPN without correction, right is with correction filter)

<img width="400" src="https://github.com/shadyabh/Correction-Filter/blob/master/figs/SR/bridge_Gauss_std1.8_x2_s.png"> <img width="400" src="https://github.com/shadyabh/Correction-Filter/blob/master/figs/SR/bridge_Gauss_std1.8_x2_s_x2_corr_corrected.png">

## Blind Super-Resolution
### Synthetic Images
Here we demonstrate the performance of our method on images that were sampled from their ground-truth image.
#### Man image from Set14
Blind super-resolution with scale factor of 4 on Gaussian model with std 4.5/sqrt(2) (left is DBPN without correction, right is with estimated correction filter)

<img width="400" src="https://github.com/shadyabh/Correction-Filter/blob/master/figs/blind_SR/man2_Gauss_std3.2_x4_s.png"> <img width="400" src="https://github.com/shadyabh/Correction-Filter/blob/master/figs/blind_SR/man2_Gauss_std3.2_x4_s_x4_corr_l0_est.png">

#### Images from DIV2KRK dataset

Blind super-resolution with scale factor of 2 tested on images from DIV2KRK dataset http://www.wisdom.weizmann.ac.il/~vision/kernelgan/ (left is DBPN without correction, right is with estimated correction filter)

<img width="400" src="https://github.com/shadyabh/Correction-Filter/blob/master/figs/blind_SR/im_31.png"> <img width="400" src="https://github.com/shadyabh/Correction-Filter/blob/master/figs/blind_SR/im_31_x2_corr_est.png">

<img width="400" src="https://github.com/shadyabh/Correction-Filter/blob/master/figs/blind_SR/im_59.png"> <img width="400" src="https://github.com/shadyabh/Correction-Filter/blob/master/figs/blind_SR/im_59_x2_corr_est.png">

<img width="400" src="https://github.com/shadyabh/Correction-Filter/blob/master/figs/blind_SR/im_66.png"> <img width="400" src="https://github.com/shadyabh/Correction-Filter/blob/master/figs/blind_SR/im_66_x2_corr_est.png">

## Real-World Super-Resolution
Here we present the results of our approach on images with no ground-truth images

### Images from Set5 dataset
Here we take images from Set5 and apply our blind SR (scale factor of 2) algorithm on them directly (without down-sampling them). 

On the left is DBPN without correction, right is with estimated correction filter.

<img width="400" src="https://github.com/shadyabh/Correction-Filter/blob/master/figs/blind_SR/bird_.png"> <img width="400" src="https://github.com/shadyabh/Correction-Filter/blob/master/figs/blind_SR/bird_x2_corr_est.png">

<img width="400" src="https://github.com/shadyabh/Correction-Filter/blob/master/figs/blind_SR/butterfly.png"> <img width="400" src="https://github.com/shadyabh/Correction-Filter/blob/master/figs/blind_SR/butterfly_x2_corr_est.png">

### Chip image

Super resolution with scale factor of 4 on the famous chip image. On the left is the original LR image, in the middle is the result of DBPN applied directly, and on the right is DBPN applied with the estimated correction filter.

<img width="80" src="https://github.com/shadyabh/Correction-Filter/blob/master/figs/blind_SR/chip_LR.png"> <img width="320" src="https://github.com/shadyabh/Correction-Filter/blob/master/figs/blind_SR/chip.png"> <img width="320" src="https://github.com/shadyabh/Correction-Filter/blob/master/figs/blind_SR/chip_x4_corrected_est.png">
