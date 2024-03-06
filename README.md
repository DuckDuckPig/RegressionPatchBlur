# Regression_Patch_Blur
This work is an expansion of Regression_Blur. For requirements and installing anaconda environment [click here](https://github.com/DuckDuckPig/Regression_Blur/tree/main)

# Training/Results file
To train your network with your own data use `VGG16_Regressor_V2.py`. The model used throughout the results and blur field prediction is under `models.py` using *VGG_14*. `Results_V2.py` uses a test dataset and accepts multiple patch sizes as a parameter to test results on different image patch sizes.

For pre-trained weights contact the author at varelal@nmsu.edu.

# Non-Uniform blur field generator
`Nub_Generator.ipynb` has the function **blur_field_gen** which generates a blur field of size *M*, *N*. This size should be the same size of the image desired to for blur. The specified *row* and *column* will divide the blur field by row patches and column patches. The values array will sweep the row patches and assign the value in the array element. For a unique values of patches there should be row * col ammount of values within the array.

The **non_uniform_blur** function will accept the sharp image *img*, and two blur field maps one for length *L* and another for angle *A*. This function will do a per pixel blur. 

# blur field prediction
`Blur_Field_Prediction.py` uses the trained model and loops through the image at a certain patch size to predict the length and angle blur parameters for the center pixel of the patch. The script does predict a per pixel blur field skipping a boarder size of *PATCH_SIZE / 2*.
