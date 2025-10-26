import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # Added for diverse ensemble
from sklearn.preprocessing import LabelEncoder
import warnings
import joblib
import os
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')

# Smart Farm Recommendation System

MODEL_DIR = 'trained_models'
os.makedirs(MODEL_DIR, exist_ok=True)

## 1. Load Data
df = pd.read_csv('Crop_recommendationV2.csv')
print("Original Data Head:")
print(df.head())

## 2. Data Cleaning and Preprocessing
print("\nData Info:")
df.info()
print("\nMissing Values:")
print(df.isnull().sum())
print("\nData Description:")
print(df.describe())

### Handle Outliers (Example: using IQR method)
# Exclude non-numeric columns like 'label' from outlier detection
numeric_df = df.select_dtypes(include=np.number)

Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1

# Identify outliers in numeric columns
outlier_mask = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)

# Filter the original DataFrame using the outlier mask
df_cleaned = df[~outlier_mask].copy() # Use .copy() to avoid SettingWithCopyWarning
print(f"\nOriginal shape: {df.shape}")
print(f"Cleaned shape (after outlier removal): {df_cleaned.shape}")

### Encode 'label' column
le = LabelEncoder()
df_cleaned['label_encoded'] = le.fit_transform(df_cleaned['label'])
print("\nLabel Encoding Applied. Unique encoded labels:")
print(df_cleaned['label_encoded'].unique())

## 3. Calculate Derived Features
# Temperature-Humidity Index (THI) - Formula is correct for Celsius, keeping as is.
df_cleaned['THI'] = df_cleaned['temperature'] - (0.55 - 0.0055 * df_cleaned['humidity']) * (df_cleaned['temperature'] - 14.5)

# Nutrient Balance Ratio (NBR) - Improved Ratio-Based NBI
# NBI = N + alpha * (P/N) + beta
alpha_nbr = 1.5
beta_nbr = 0.5
# Handle division by zero for NBR: if N is 0, P/N is 0.
df_cleaned['P_div_N'] = df_cleaned.apply(lambda row: row['P'] / row['N'] if row['N'] != 0 else 0, axis=1)
df_cleaned['NBR'] = df_cleaned['N'] + alpha_nbr * df_cleaned['P_div_N'] + beta_nbr
df_cleaned.drop('P_div_N', axis=1, inplace=True) # Drop the temporary column

# Water Availability Index (WAI) - Water Balance Index using simplified PET (Hargreaves)
# PET = 0.0023 * Ra * (T_mean + 17.8) * (T_max - T_min)**0.5
# Approximations: T_mean = temperature, T_max = temperature + 5, T_min = temperature - 5, Ra = 15 (average extraterrestrial radiation)
Ra = 15 # MJ/m^2/day, a constant average for simplicity
T_mean = df_cleaned['temperature']
T_max = df_cleaned['temperature'] + 5
T_min = df_cleaned['temperature'] - 5
# Ensure (T_max - T_min) is non-negative for sqrt
temp_diff_sqrt = np.sqrt(np.maximum(0, T_max - T_min))
df_cleaned['PET'] = 0.0023 * Ra * (T_mean + 17.8) * temp_diff_sqrt
df_cleaned['WAI'] = df_cleaned['soil_moisture'] + df_cleaned['rainfall'] - df_cleaned['PET']

# Photosynthesis Potential (PP) - Simplified proxy, keeping as is based on user's note.
df_cleaned['PP'] = df_cleaned['sunlight_exposure'] * df_cleaned['co2_concentration'] * df_cleaned['temperature']

# Soil Fertility Index (SFI) - Improved with pH Penalty
# SFI = Organic_Matter * NBR * f(pH)
# f(pH) = 1 - (|pH - 6.5| / 6.5)^gamma
optimal_pH = 6.5
gamma_sfi = 2
# Calculate pH stress factor, ensuring absolute difference is used
pH_stress_factor = 1 - (np.abs(df_cleaned['ph'] - optimal_pH) / optimal_pH)**gamma_sfi
df_cleaned['SFI'] = df_cleaned['organic_matter'] * df_cleaned['NBR'] * pH_stress_factor

print("\nData Head with Derived Features:")
print(df_cleaned.head())

## 4. Feature Scaling and PCA (for predicting THI)
# We will predict all derived features as target variables
target_indicators = ['THI', 'NBR', 'WAI', 'PP', 'SFI']

# Define X and y here so they are always available
X = df_cleaned.drop(['label'] + target_indicators, axis=1) # Drop original label and all derived features
y = df_cleaned[target_indicators] # Target is now all derived indicators

# Get all feature columns used for training (before scaling/PCA)
# This needs to be consistent whether models are loaded or trained
feature_columns_for_prediction = X.columns.tolist()

# Try loading models and preprocessing objects
scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
pca_path = os.path.join(MODEL_DIR, 'pca.pkl')
le_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')

dl_ensemble_models = []
rf_ensemble_models = []
gb_ensemble_models = []

models_loaded = False
try:
    print("\nAttempting to load trained models and preprocessing objects...")
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    le = joblib.load(le_path)

    # Load DL models
    for i in range(3): # Assuming 3 DL models
        dl_ensemble_models.append(load_model(os.path.join(MODEL_DIR, f'dl_model_{i}.keras')))
    
    # Load RF models
    for i in range(len(target_indicators)):
        rf_ensemble_models.append(joblib.load(os.path.join(MODEL_DIR, f'rf_model_{i}.pkl')))

    # Load GB models
    for i in range(len(target_indicators)):
        gb_ensemble_models.append(joblib.load(os.path.join(MODEL_DIR, f'gb_model_{i}.pkl')))
    
    models_loaded = True
    print("Models and preprocessing objects loaded successfully. Skipping training.")

except FileNotFoundError:
    print("Trained models not found. Proceeding with training...")
    models_loaded = False
except Exception as e:
    print(f"Error loading models: {e}. Proceeding with training.")
    models_loaded = False

if not models_loaded:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.95) # Retain 95% of variance
    X_pca = pca.fit_transform(X_scaled)

    print(f"\nOriginal number of features: {X.shape[1]}")
    print(f"Number of features after PCA: {X_pca.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # Function to create a base Deep Learning model
    def create_base_dl_model(input_dim, output_dim):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(output_dim, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
        return model

    # To store predictions on X_test for overall ensemble evaluation
    all_model_test_predictions = []

    print("\nTraining Diverse Models for Ensemble...")

    # 1. Deep Learning Models (3 instances)
    num_dl_models = 3
    dl_test_predictions = []
    for i in range(num_dl_models):
        print(f"Training Deep Learning Model {i+1}/{num_dl_models}...")
        dl_model = create_base_dl_model(X_train.shape[1], len(target_indicators))
        dl_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
        
        loss, mae = dl_model.evaluate(X_test, y_test, verbose=0)
        print(f"DL Model {i+1} - Mean Squared Error: {loss:.4f}, Mean Absolute Error: {mae:.4f}")
        
        dl_ensemble_models.append(dl_model)
        dl_test_predictions.append(dl_model.predict(X_test))

    # Average DL model predictions for ensemble
    if dl_test_predictions:
        avg_dl_test_predictions = np.mean(dl_test_predictions, axis=0)
        all_model_test_predictions.append(avg_dl_test_predictions)

    # 2. Random Forest Regressor (one model per target indicator)
    print("\nTraining Random Forest Regressor (multi-output)...")
    rf_preds_test_individual = []
    for i, indicator in enumerate(target_indicators):
        print(f"  Training Random Forest for {indicator}...")
        rf_model_single = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model_single.fit(X_train, y_train[indicator])
        rf_preds_test_individual.append(rf_model_single.predict(X_test))
        rf_ensemble_models.append(rf_model_single)

    rf_pred_test_combined = np.column_stack(rf_preds_test_individual)
    rf_loss = np.mean((rf_pred_test_combined - y_test.values)**2)
    rf_mae = np.mean(np.abs(rf_pred_test_combined - y_test.values))
    print(f"Random Forest Ensemble - Mean Squared Error: {rf_loss:.4f}, Mean Absolute Error: {rf_mae:.4f}")
    all_model_test_predictions.append(rf_pred_test_combined)

    # 3. Gradient Boosting Regressor (one model per target indicator)
    print("\nTraining Gradient Boosting Regressor (multi-output)...")
    gb_preds_test_individual = []
    for i, indicator in enumerate(target_indicators):
        print(f"  Training Gradient Boosting for {indicator}...")
        gb_model_single = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model_single.fit(X_train, y_train[indicator])
        gb_preds_test_individual.append(gb_model_single.predict(X_test))
        gb_ensemble_models.append(gb_model_single)

    gb_pred_test_combined = np.column_stack(gb_preds_test_individual)
    gb_loss = np.mean((gb_pred_test_combined - y_test.values)**2)
    gb_mae = np.mean(np.abs(gb_pred_test_combined - y_test.values))
    print(f"Gradient Boosting Ensemble - Mean Squared Error: {gb_loss:.4f}, Mean Absolute Error: {gb_mae:.4f}")
    all_model_test_predictions.append(gb_pred_test_combined)

    # Evaluate the overall ensemble model using the averaged predictions
    y_pred_overall_ensemble = np.mean(all_model_test_predictions, axis=0)
    overall_ensemble_loss = np.mean((y_pred_overall_ensemble - y_test.values)**2)
    overall_ensemble_mae = np.mean(np.abs(y_pred_overall_ensemble - y_test.values))

    print(f"\nOverall Ensemble Model Mean Squared Error (averaged predictions): {overall_ensemble_loss:.4f}")
    print(f"Overall Ensemble Model Mean Absolute Error (averaged predictions): {overall_ensemble_mae:.4f}")

## 6. Save Trained Models and Preprocessing Objects
print("\nSaving trained models and preprocessing objects...")
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
joblib.dump(pca, os.path.join(MODEL_DIR, 'pca.pkl'))
joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl')) # Save label encoder as well

for i, dl_model in enumerate(dl_ensemble_models):
    dl_model.save(os.path.join(MODEL_DIR, f'dl_model_{i}.keras')) # Keras models saved in .keras format

for i, rf_model in enumerate(rf_ensemble_models):
    joblib.dump(rf_model, os.path.join(MODEL_DIR, f'rf_model_{i}.pkl'))

for i, gb_model in enumerate(gb_ensemble_models):
    joblib.dump(gb_model, os.path.join(MODEL_DIR, f'gb_model_{i}.pkl'))
print("Models and preprocessing objects saved successfully.")

## 7. Calculate Crop-Specific Ideal Indicator Ranges
# Calculate the 25th and 75th percentiles for each indicator, grouped by crop label.
crop_ideal_indicator_ranges = {}
for crop_label in df_cleaned['label'].unique():
    crop_data = df_cleaned[df_cleaned['label'] == crop_label]
    ranges = {}
    for indicator in target_indicators:
        q1 = crop_data[indicator].quantile(0.25)
        q3 = crop_data[indicator].quantile(0.75)
        ranges[indicator] = (q1, q3)
    crop_ideal_indicator_ranges[crop_label] = ranges

print("\nCalculated Crop-Specific Ideal Indicator Ranges (25th-75th percentile):")
for crop, ranges in crop_ideal_indicator_ranges.items():
    print(f"  {crop}:")
    for indicator, (min_val, max_val) in ranges.items():
        print(f"    {indicator}: ({min_val:.2f}, {max_val:.2f})")

## 7. Predictive Analysis and Inverse Recommendation System

# The goal is to predict multiple indicators and then, given ideal indicator values,
# suggest changes to sensor values.

### Example Usage of Prediction Function for Multiple Indicators
def predict_indicators_ensemble(input_data, scaler_obj, pca_obj, dl_models, rf_models, gb_models, feature_columns, target_indicators_list):
    # Ensure input_data is a DataFrame with correct columns
    input_df = pd.DataFrame([input_data], columns=feature_columns)

    # Scale the input data
    input_scaled = scaler_obj.transform(input_df)

    # Apply PCA
    input_pca = pca_obj.transform(input_scaled)

    all_individual_predictions = []

    # Get predictions from DL models
    dl_individual_predictions = [model.predict(input_pca)[0] for model in dl_models]
    if dl_individual_predictions:
        all_individual_predictions.append(np.mean(dl_individual_predictions, axis=0))

    # Get predictions from Random Forest models
    rf_individual_predictions = [model.predict(input_pca)[0] for model in rf_models]
    if rf_individual_predictions:
        all_individual_predictions.append(np.column_stack(rf_individual_predictions)[0])

    # Get predictions from Gradient Boosting models
    gb_individual_predictions = [model.predict(input_pca)[0] for model in gb_models]
    if gb_individual_predictions:
        all_individual_predictions.append(np.column_stack(gb_individual_predictions)[0])
    
    # Average the predictions from all model types
    ensembled_predictions = np.mean(all_individual_predictions, axis=0)
    
    return dict(zip(target_indicators_list, ensembled_predictions))

# Get all feature columns used for training
feature_columns_for_prediction = X.columns.tolist()

# Create a sample input based on a random entry from the dataset
sample_entry = df_cleaned.drop(['label'] + target_indicators, axis=1).sample(1, random_state=42).iloc[0].to_dict()

# Predict all indicators for the sample input using the ensemble
predicted_indicators = predict_indicators_ensemble(sample_entry, scaler, pca, dl_ensemble_models, rf_ensemble_models, gb_ensemble_models, feature_columns_for_prediction, target_indicators)
print(f"\nSample Input: {sample_entry}")
print(f"Predicted Indicators (Ensemble): {predicted_indicators}")

### Inverse Recommendation Function (Conceptual - requires more advanced techniques)
# This is a placeholder for a function that would take desired indicator values
# and suggest changes to input sensor values. This is a complex inverse problem
# that typically requires optimization, reinforcement learning, or a separate
# inverse model.

# For a simplified approach, we can define "ideal" ranges for indicators
# and then suggest adjustments to sensor values based on deviations.
# This would be similar to the original `recommend_actions` but for indicators.

# Let's define a simple function to suggest sensor adjustments based on predicted indicators
# and some hypothetical ideal ranges.

def recommend_sensor_changes_from_indicators(predicted_indicators_dict, crop_label, crop_ideal_ranges_dict):
    recommendations = []
    if crop_label not in crop_ideal_ranges_dict:
        recommendations.append(f"No ideal ranges defined for crop: {crop_label}. Using general recommendations.")
        # Fallback to a general range if specific not found, or handle as an error
        # For now, if no specific range, we'll just note it.
        ideal_ranges = {} # No fallback to arbitrary general ranges
    else:
        ideal_ranges = crop_ideal_ranges_dict[crop_label]

    for indicator, predicted_value in predicted_indicators_dict.items():
        if indicator in ideal_ranges:
            ideal_min, ideal_max = ideal_ranges[indicator]
            if predicted_value < ideal_min:
                recommendations.append(f"For {crop_label}, increase conditions affecting {indicator} (e.g., adjust temperature, humidity, NPK). Predicted {indicator}: {predicted_value:.2f}, Ideal: {ideal_min:.2f}-{ideal_max:.2f}")
            elif predicted_value > ideal_max:
                recommendations.append(f"For {crop_label}, decrease conditions affecting {indicator} (e.g., adjust temperature, humidity, NPK). Predicted {indicator}: {predicted_value:.2f}, Ideal: {ideal_min:.2f}-{ideal_max:.2f}")
            else:
                recommendations.append(f"For {crop_label}, {indicator} is within ideal range. Predicted: {predicted_value:.2f}, Ideal: {ideal_min:.2f}-{ideal_max:.2f}")
        else:
            recommendations.append(f"No ideal range defined for {indicator} for crop {crop_label}.")
    return recommendations
