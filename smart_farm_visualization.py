import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

# This file contains data visualization and analysis components for the Smart Farm Recommendation System.

MODEL_DIR = 'trained_models'

def calculate_derived_features(df):
    """
    Calculates derived features (THI, NBR, WAI, PP, SFI) for the DataFrame.
    This function is duplicated from smart_farm_recommendation.py to ensure consistency.
    """
    df_copy = df.copy() # Work on a copy to avoid modifying original df

    # Temperature-Humidity Index (THI)
    df_copy['THI'] = df_copy['temperature'] - (0.55 - 0.0055 * df_copy['humidity']) * (df_copy['temperature'] - 14.5)

    # Nutrient Balance Ratio (NBR)
    alpha_nbr = 1.5
    beta_nbr = 0.5
    df_copy['P_div_N'] = df_copy.apply(lambda row: row['P'] / row['N'] if row['N'] != 0 else 0, axis=1)
    df_copy['NBR'] = df_copy['N'] + alpha_nbr * df_copy['P_div_N'] + beta_nbr
    df_copy.drop('P_div_N', axis=1, inplace=True)

    # Water Availability Index (WAI)
    Ra = 15
    T_mean = df_copy['temperature']
    T_max = df_copy['temperature'] + 5
    T_min = df_copy['temperature'] - 5
    temp_diff_sqrt = np.sqrt(np.maximum(0, T_max - T_min))
    df_copy['PET'] = 0.0023 * Ra * (T_mean + 17.8) * temp_diff_sqrt
    df_copy['WAI'] = df_copy['soil_moisture'] + df_copy['rainfall'] - df_copy['PET']
    # df_copy.drop('PET', axis=1, inplace=True) # PET is kept as a feature for consistency with recommendation script

    # Photosynthesis Potential (PP)
    df_copy['PP'] = df_copy['sunlight_exposure'] * df_copy['co2_concentration'] * df_copy['temperature']

    # Soil Fertility Index (SFI)
    optimal_pH = 6.5
    gamma_sfi = 2
    pH_stress_factor = 1 - (np.abs(df_copy['ph'] - optimal_pH) / optimal_pH)**gamma_sfi
    df_copy['SFI'] = df_copy['organic_matter'] * df_copy['NBR'] * pH_stress_factor
    
    return df_copy

def perform_visualization_and_analysis(df_cleaned_with_derived, X_pca, pca_obj, le_obj):
    """
    Performs data visualization and PCA analysis, including derived features.

    Args:
        df_cleaned_with_derived (pd.DataFrame): The cleaned DataFrame with derived features.
        X_pca (np.ndarray): PCA-transformed features.
        pca_obj (PCA): The fitted PCA object.
        le_obj (LabelEncoder): The fitted LabelEncoder object.
    """
    print("\n--- Starting Data Visualization ---")

    # Correlation Matrix of All Features (Original + Derived)
    plt.figure(figsize=(18, 14)) # Increased figure size for better readability
    sns.heatmap(df_cleaned_with_derived.drop('label', axis=1).corr(), annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8}) # Added fmt and annot_kws
    plt.title('Correlation Matrix of All Features (Original + Derived)')
    plt.show()

    print("\n--- Starting PCA Analysis ---")

    ## PCA Analysis
    print(f"\nOriginal number of features (before PCA): {df_cleaned_with_derived.drop('label', axis=1).shape[1]}")
    print(f"Number of features after PCA (retaining 95% variance): {pca_obj.n_components_}")

    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca_obj.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA - Cumulative Explained Variance')
    plt.grid(True)
    plt.show()

    # Visualize PCA components (first two components)
    if X_pca.shape[1] >= 2:
        pca_df = pd.DataFrame(data=X_pca[:, :2], columns=['principal component 1', 'principal component 2'])
        pca_df['label'] = le_obj.inverse_transform(df_cleaned_with_derived['label_encoded']) # Use inverse transform
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='principal component 1', y='principal component 2', hue='label', data=pca_df, palette='viridis', s=100, alpha=0.7)
        plt.title('2 Component PCA')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)
        plt.show()
    else:
        print("Not enough PCA components to visualize 2D scatter plot.")

    print("\n--- Data Visualization and Analysis Complete ---")

if __name__ == '__main__':
    # Load data
    df = pd.read_csv('Crop_recommendationV2.csv')
    
    # Data Cleaning and Preprocessing (consistent with smart_farm_recommendation.py)
    numeric_df = df.select_dtypes(include=np.number)
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
    df_cleaned = df[~outlier_mask].copy()

    # Encode 'label' column
    le = LabelEncoder()
    df_cleaned['label_encoded'] = le.fit_transform(df_cleaned['label'])
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl')) # Save for consistency

    # Calculate Derived Features
    df_cleaned_with_derived = calculate_derived_features(df_cleaned)

    # Prepare data for PCA (consistent with smart_farm_recommendation.py)
    target_indicators = ['THI', 'NBR', 'WAI', 'PP', 'SFI']
    # X should contain 'label_encoded' and 'PET' as per smart_farm_recommendation.py
    X = df_cleaned_with_derived.drop(['label'] + target_indicators, axis=1)

    # Load or fit scaler and PCA
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    pca_path = os.path.join(MODEL_DIR, 'pca.pkl')
    le_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')

    scaler = None
    pca_obj = None
    le_obj = None

    try:
        print("\nAttempting to load scaler, PCA, and label encoder objects...")
        scaler = joblib.load(scaler_path)
        pca_obj = joblib.load(pca_path)
        le_obj = joblib.load(le_path)
        print("Scaler, PCA, and label encoder objects loaded successfully.")
    except FileNotFoundError:
        print("Preprocessing objects not found. Fitting new ones...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca_obj = PCA(n_components=0.95)
        X_pca = pca_obj.fit_transform(X_scaled)
        joblib.dump(scaler, scaler_path)
        joblib.dump(pca_obj, pca_path)
        le_obj = le # Use the newly fitted label encoder
        print("New scaler, PCA, and label encoder objects fitted and saved.")
    except Exception as e:
        print(f"Error loading preprocessing objects: {e}. Fitting new ones...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca_obj = PCA(n_components=0.95)
        X_pca = pca_obj.fit_transform(X_scaled)
        joblib.dump(scaler, scaler_path)
        joblib.dump(pca_obj, pca_path)
        le_obj = le # Use the newly fitted label encoder
        print("New scaler, PCA, and label encoder objects fitted and saved.")

    # If objects were loaded, transform X
    if scaler is not None and pca_obj is not None:
        X_scaled = scaler.transform(X)
        X_pca = pca_obj.transform(X_scaled)
    else:
        # This case should be handled by the 'fitting new ones' block above
        pass

    perform_visualization_and_analysis(df_cleaned_with_derived, X_pca, pca_obj, le_obj)
