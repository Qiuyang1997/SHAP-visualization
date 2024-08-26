import rasterio
import pandas as pd
import numpy as np
import lightgbm as lgb  # Alternative machine learning models are available
import shap as shap

# input dataset
data = pd.read_excel('data of nitrate')
y = data["concentration of nitrate"]
X = data.drop(["concentration of nitrate"], axis=1)

# Set parameter space
params = {}
# model fit
model = lgb.LGBMRegressor(**params) # Alternative machine learning models are available
model.fit(X, y)

# Raster file path list, and build a prediction data set
input_raster_filepaths = []  # Enter the raster file names in square brackets here
                             # The raster format needs to be (.tif) and the reference frame and resolution needs to be uniform in the GIS software
raster_data = []
profiles = []
for file_path in input_raster_filepaths:
    with rasterio.open(file_path) as src:
	    raster_data.append(src.read(1).astype(np.float32))
	    profiles.append(src.profile)
	 # Coordinate transformation
profile = profiles[0]
transform = profile["transform"]
	 # Stack raster data into multidimensional arrays
stacked_raster = np.stack(raster_data, axis=-1)
rows, cols, num_bands = stacked_raster.shape
	 # Converts raster data to 2D array form
reshaped_raster = stacked_raster.reshape((rows * cols, num_bands))

# SHAP and visualization
tree_explainer = shap.TreeExplainer(model) #  ML model that not based on decision tree should use “shap.Explainer” to replace “shap.TreeExplainer”
shap_values = tree_explainer.shap_values(reshaped_raster)  # ML model that not based on decision tree should use “explainer” to replace “tree_explainer”
     # Output SAHP values as Raster file
for i, input_raster_filepath in enumerate(input_raster_filepaths):
	output_filepath = 'output file' + input_raster_filepath.split('\\')[-1].split('.')[0] + '_SHAPvalues.tif'  #  Set the location and name of the output raster file
	shap_raster = shap_values[:, i].reshape((rows, cols))  # Get the SHAP value grid for a particular grid
	profile.update(count=1, dtype=rasterio.float32)   #  Set the numeric type of the output raster file
	with rasterio.open(output_filepath, 'w', **profile) as dst:
	    dst.write(shap_raster, 1)
{}

#################################    Additional considerations    ########################################
"""
1. This code is also applicable to the classification algorithm, which only needs to set in advance whether the output SHAP values
   correspond to the marginal contribution of positive (1) or negative (0), and usually some algorithms will default to positive.
   For example can output SAHP values for class1 as Raster file by changing code line 37-38 as follows:
tree_explainer = shap.TreeExplainer(model)
shap_values = tree_explainer.shap_values(reshaped_raster)
shap_values_class_1 = shap_values[1]
																																
2. However, algorithms that use the ensemble techniques to solve the data imbalance (e.g., Easyensemble, BalanceCascade, UnderOverBagging, and BalanceRandomForestClassifier)
   cannot call the SHAP library directly, which should calculate the marginal contribution of the metrics directly from game theory and will add a lot of effort.

3. Since the original codes of 'lightgbm', 'numpy', 'shap', and 'rasterio' comes from different authors and their version updates are not synchronized, 
   so it's necessary to keep their versions in a "delicate balance" when calling these libraries to avoid unnecessary trouble