"""
CropHarvest Data Processing and Classification Script

This script processes satellite imagery data from the CropHarvest dataset to classify cropland areas.
It computes the Variance of Second Derivative (var2der) complexity metric for selected spectral bands
and uses these features to train both CatBoost and Random Forest classifiers. The script excludes
specific bands and evaluates the models using various metrics including F1 Score, AUC-ROC, and Accuracy.

**Spectral Bands and Features:**

1. **VV (Vertical-Vertical) and VH (Vertical-Horizontal):**
   - **Type:** Synthetic Aperture Radar (SAR) Polarization Bands
   - **Description:** VV and VH bands capture backscatter from the Earth's surface, providing information
     about surface roughness and moisture content. They are useful for analyzing soil properties and
     vegetation structure.

2. **B2 (Blue), B3 (Green), B4 (Red), B5 (Red Edge 1), B6 (Red Edge 2), B7 (Red Edge 3):**
   - **Type:** Sentinel-2 Optical Bands
   - **Description:** These bands cover the visible to near-infrared spectrum, capturing detailed information
     about vegetation health, chlorophyll content, and water absorption features.

3. **B8 (NIR - Near Infrared) and B8A (Narrow NIR):**
   - **Type:** Sentinel-2 Near-Infrared Bands
   - **Description:** B8 and B8A bands are crucial for vegetation analysis, as they are sensitive to plant
     health and biomass. They are used in calculating indices like NDVI.

4. **B9 (Water Vapor), B11 (SWIR - Short-Wave Infrared 1), B12 (SWIR - Short-Wave Infrared 2):**
   - **Type:** Sentinel-2 Infrared Bands
   - **Description:** These bands provide information about atmospheric water vapor, soil moisture, and
     vegetation water content. They are essential for assessing plant stress and water availability.

5. **NDVI (Normalized Difference Vegetation Index):**
   - **Type:** Vegetation Index
   - **Description:** NDVI is calculated using the Red and NIR bands to assess vegetation health and density.
     It ranges from -1 to +1, where higher values indicate healthier and denser vegetation.

6. **temperature_2m:**
   - **Type:** Meteorological Feature
   - **Description:** Represents the surface temperature at 2 meters above ground. It's useful for understanding
     the thermal properties of the land surface, which can affect vegetation growth.

7. **total_precipitation:**
   - **Type:** Meteorological Feature
   - **Description:** Indicates the total precipitation received in the area. Precipitation levels influence soil moisture
     and plant growth.

8. **elevation:**
   - **Type:** Topographical Feature
   - **Description:** Represents the elevation of the terrain. Elevation affects climate conditions, soil types, and
     vegetation distribution.

9. **slope:**
   - **Type:** Topographical Feature
   - **Description:** Measures the steepness of the terrain. Slope influences water runoff, soil erosion, and
     suitability for certain crops.

**Script Workflow:**

1. **Data Loading:**
   - Loads labels and time series data from H5 files.
   - Selects a subset of the data for processing.

2. **Feature Extraction:**
   - Excludes specified bands (`'total_precipitation'`, `'elevation'`, `'slope'`).
   - Computes the `var2der` metric for each selected band.
   - Constructs a feature matrix for model training.

3. **Model Training and Evaluation:**
   - Trains a CatBoost classifier on the extracted features.
   - Trains a Random Forest classifier on the CropHarvest benchmark dataset (Togo dataset) as part of CropHarvest tutorials.
   - Evaluates both models using Classification Report, F1 Score, AUC-ROC, and Accuracy.
   - Analyzes feature importance.

4. **Inference:**
   - Downloads a test file for inference.
   - Runs inference using the trained models.
   - Visualizes the prediction results.

**Usage:**
Ensure that the `CropHarvestLabels`, `CropHarvest`, `Inference`, `func_complexity_metrics`, and required libraries are properly installed and configured.
Run the script in an environment where the CropHarvest dataset is accessible.

**Note:**
- The script assumes that the `BANDS` list from `cropharvest.bands` includes all the necessary spectral bands.
- Adjust the `embedding_dimension` and `time_delay` parameters as needed based on the characteristics of your time series data.
"""

# Import necessary libraries
from cropharvest.datasets import CropHarvestLabels, CropHarvest
from cropharvest.inference import Inference
import h5py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import requests
import tempfile

# Import the function from func_complexity_metrics.py
from func_complexity_metrics import calculate_variance_2nd_derivative
from cropharvest.bands import BANDS

# Import classifiers and metrics
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score

# Import necessary libraries
from cropharvest.datasets import CropHarvestLabels
import os

# Define data directory
DATA_DIR = "data_austria"

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Load evaluation datasets
evaluation_datasets = CropHarvest.create_benchmark_datasets(DATA_DIR, download=True)

# Load labels
labels = CropHarvestLabels(DATA_DIR, download=True)
labels_geojson = labels.as_geojson()

# Define bounding boxes for Austria and Germany
AUSTRIA_BBOX_old = {
    'lat_min': 47.2,
    'lat_max': 49.0,
    'lon_min': 9.5,
    'lon_max': 17.0
}

AUSTRIA_BBOX = {
    'lat_min': 47.0,
    'lat_max': 55.0,
    'lon_min': 5.9,
    'lon_max': 17.0
}

GERMANY_BBOX = {
    'lat_min': 47.0,
    'lat_max': 55.0,
    'lon_min': 5.9,
    'lon_max': 15.0
}

# Filter labels for Austria
subset_labels = labels_geojson[
    (labels_geojson['lat'] >= AUSTRIA_BBOX['lat_min']) &
    (labels_geojson['lat'] <= AUSTRIA_BBOX['lat_max']) &
    (labels_geojson['lon'] >= AUSTRIA_BBOX['lon_min']) &
    (labels_geojson['lon'] <= AUSTRIA_BBOX['lon_max'])
]

if subset_labels.empty:
    print("No data available for Austria. Trying Germany...")
    # Filter labels for Germany
    subset_labels = labels_geojson[
        (labels_geojson['lat'] >= GERMANY_BBOX['lat_min']) &
        (labels_geojson['lat'] <= GERMANY_BBOX['lat_max']) &
        (labels_geojson['lon'] >= GERMANY_BBOX['lon_min']) &
        (labels_geojson['lon'] <= GERMANY_BBOX['lon_max'])
    ]

if subset_labels.empty:
    raise ValueError("No data available for Austria or Germany.")

print(f"Processing {len(subset_labels)} samples from the selected region.")

# Initialize lists to store data
time_series_data = []
labels_list = []

print(subset_labels)

# Iterate over the subset to extract time series data and labels
for index, row in subset_labels.iterrows():
    # Get the path to the h5 file containing the time series data
    path_to_file = labels._path_from_row(row)
    if not os.path.exists(path_to_file):
        print('File does not exist')
        continue  # Skip if the file does not exist

    # **Extract the label using the correct key**
    label = row.get('is_crop')
    #print(label)

    if label is None:
        print(f"No label found for index {index}, skipping.")
        continue

    # Open the h5 file and extract the time series array
    with h5py.File(path_to_file, "r") as h5_file:
        x = h5_file.get("array")[:]
        time_series_data.append(x)
        labels_list.append(label)


# List of bands to skip
bands_to_skip = ['total_precipitation', 'elevation', 'slope', 'B4', 'B5', 'B7', 'B8A', 'B8', 'B6']
#bands_to_skip = ['total_precipitation', 'elevation', 'slope', 'B4', 'B5', 'B8A', 'B8']
selected_bands = [band for band in BANDS if band not in bands_to_skip]

print(f"Selected bands for processing: {selected_bands}")

# Initialize a dictionary to store var2der features for each selected band
var2der_features = {f"{band}_var2der": [] for band in selected_bands}
valid_labels_list = []

# Parameters for the functions
embedding_dimension = 3  # Adjust as needed
time_delay = 1
make_square = False  # As per the function definitions

# Iterate over the time series data to compute var2der for each band
for idx, x in enumerate(time_series_data):
    print(f"Processing sample index: {idx}")

    # Initialize a dictionary to hold var2der for the current sample
    current_var2der = {}
    skip_sample = False  # Flag to determine if the sample should be skipped

    for band in selected_bands:
        try:
            band_index = BANDS.index(band)
        except ValueError:
            print(f"Band '{band}' not found in BANDS list. Skipping this band.")
            current_var2der[f"{band}_var2der"] = 0
            continue

        # Extract the band's time series
        band_series = x[:, band_index]

        # Replace NaN or infinite values with zeros
        band_series = np.nan_to_num(band_series, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure the time series has sufficient length
        required_length = embedding_dimension + (embedding_dimension - 1) * time_delay
        if len(band_series) < required_length:
            print(f"Skipping sample at index {idx} due to insufficient time series length for band '{band}'.")
            skip_sample = True
            break  # No need to process further bands for this sample

        # Compute Variance of Second Derivative (var2der) for the band
        try:
            var2der = calculate_variance_2nd_derivative(
                band_series,
                embedding_dimension=embedding_dimension,
                time_delay=time_delay
            )
        except Exception as e:
            print(f"{band} var2der computation failed at index {idx}: {e}")
            var2der = 0  # Assign 0 if computation fails

        current_var2der[f"{band}_var2der"] = var2der

    if skip_sample:
        continue  # Skip appending features and label for this sample

    # Append the var2der features for all selected bands
    for feature_name, value in current_var2der.items():
        var2der_features[feature_name].append(value)

    # Append the label for this sample
    valid_labels_list.append(labels_list[idx])


# Ensure that all var2der feature lists have the same length as the labels list
feature_lengths = [len(v) for v in var2der_features.values()]
if not all(length == len(valid_labels_list) for length in feature_lengths):
    raise ValueError("Mismatch between feature lengths and label list length.")

# Create a DataFrame with the var2der features and labels
features_df = pd.DataFrame(var2der_features)
features_df['label'] = valid_labels_list

# Drop any rows with NaN values (if any)
features_df = features_df.dropna()

# Check if the DataFrame is empty
if features_df.empty:
    print("The features DataFrame is empty. Cannot proceed with training.")
    exit()

# Preview the features DataFrame
print("Features DataFrame:")
print(features_df.head())
print(f"Total samples in features_df: {len(features_df)}")

# Check label distribution
print("Label distribution:")
print(features_df['label'].value_counts())

# ----------------------------
# CatBoost Classifier
# ----------------------------

# Separate features and labels
X_features_cb = features_df.drop('label', axis=1)
y_features_cb = features_df['label']

# Split the data into training and testing sets
X_train_cb, X_test_cb, y_train_cb, y_test_cb = train_test_split(
    X_features_cb, y_features_cb, test_size=0.2, random_state=42
)

# Initialize CatBoost Classifier
cat_model = CatBoostClassifier(
    iterations=100, learning_rate=0.1, verbose=0, random_state=42
)

# Train the model
cat_model.fit(X_train_cb, y_train_cb)

# Make predictions
y_pred_cb = cat_model.predict(X_test_cb)
y_pred_proba_cb = cat_model.predict_proba(X_test_cb)[:,1]

# Evaluate the model
print("CatBoost Classification Report:")
print(classification_report(y_test_cb, y_pred_cb))

# Compute additional metrics
f1_cb = f1_score(y_test_cb, y_pred_cb)
auc_cb = roc_auc_score(y_test_cb, y_pred_proba_cb)
accuracy_cb = accuracy_score(y_test_cb, y_pred_cb)

print("CatBoost Evaluation Metrics:")
print(f"F1 Score: {f1_cb}")
print(f"AUC-ROC: {auc_cb}")
print(f"Accuracy: {accuracy_cb}")

# Analyze feature importance for CatBoost
# Get feature importances
feature_importances_cb = cat_model.get_feature_importance()
feature_names_cb = X_features_cb.columns

processed_feature_names_cb = ["complexity " + (name.replace("_var2der", "")).replace("_2m", "") for name in feature_names_cb]

# Create a DataFrame for better visualization
fi_df_cb = pd.DataFrame({
    'Feature': processed_feature_names_cb,
    'Importance': feature_importances_cb
}).sort_values(by='Importance', ascending=False)

# Define font sizes
LABEL_FONT_SIZE = 22.5
TITLE_FONT_SIZE = 22.5
TICK_FONT_SIZE = 22.5
LEGEND_FONT_SIZE = 22.5

# Plot the feature importances with customized aesthetics
plt.figure(figsize=(14, 10))
plt.barh(fi_df_cb['Feature'], fi_df_cb['Importance'], color='#5ba234')
plt.xlabel('Importance', fontsize=LABEL_FONT_SIZE, fontweight='bold')
plt.title('Feature Importances', fontsize=TITLE_FONT_SIZE, fontweight='bold')
plt.gca().invert_yaxis()

# Customize tick parameters
plt.xticks(fontsize=TICK_FONT_SIZE)
plt.yticks(fontsize=TICK_FONT_SIZE)

# Add grid
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Save the figure with specified aesthetics
plt.savefig(
    'catboost_feature_importances.png',
    transparent=True,
    bbox_inches='tight',
    pad_inches=0.1,
    dpi=300
)

# Display the plot
plt.show()

# Display the Feature Importance DataFrame for CatBoost
print(fi_df_cb)

