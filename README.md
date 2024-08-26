# SHAP-visualization
Convert SHAP analysis results into raster files readable in GIS environment using python
The process of spatially visualizing SHAP values in a GIS environment involves four steps, and the code for these steps is provided in the Supplementary material. In this study, the code was executed in a Python 3.6 environment (Table 2):
1. Preprocess input training data: Include nitrate concentration and values of corresponding influencing factors. Train the ML model using this data.
2. Preprocess raster data: In GIS software (this study utilized ArcGIS 10.6 software), convert raster data to .TIF format, and unify their coordinate system and resolution. Expand raster layers to 2D arrays to use as the prediction dataset.
3. Calculate SHAP values: Different machine learning models require different SHAP statements. We recommend the LightGBM model for its efficiency with large datasets and its use of TreeSHAP, an optimized SHAP method.
4. Convert SHAP values to raster format: Transform SHAP values from 2D arrays to raster format. Execute the provided code in a Python environment to visualize the spatial distribution of SHAP values in the GIS software (e.g. ArcGIS and QGIS).
