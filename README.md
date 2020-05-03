# Correction-Filter

Correction Filter for Single Image Super-Resolution: Robustifying Off-the-Shelf Deep Super-Resolvers (https://arxiv.org/abs/1912.00157 , Accepted to CVPR 2020 - oral)

# Non-Blind
1. Change the out/in directories in "correct_img.py".
2. Define the sampling/reconstruction basis (s and r) in "correct_img.py".
3. Set the scale factor in "Config.py".
4. Run "correct_img.py"

# Blind
1. Define the SR network in "estimate_correction.py".
2. Update the out/in directories in "estimate_correction.py"
3. Set the scale factor in "Config.py".
4. Run "estimate_correction.py"

# Citation:

    @ARTICLE{correction_filter,
      author = {{Abu Hussein}, Shady and {Tirer}, Tom and {Giryes}, Raja},
      title = "{Correction Filter for Single Image Super-Resolution: Robustifying Off-the-Shelf Deep Super-Resolvers}",
      journal = {In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      year = "2020"
    }
