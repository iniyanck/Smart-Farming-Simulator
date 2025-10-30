import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
import joblib
import os
from typing import Dict, List, Any, Optional, Union

warnings.filterwarnings('ignore')

MODEL_DIR = 'trained_models'
os.makedirs(MODEL_DIR, exist_ok=True)

class RecommendationSystem:
    """
    Manages the machine learning models and logic for crop recommendation.
    Handles data preprocessing, model training, prediction of indicators,
    and generation of structured recommendations for sensor adjustments.
    """
    def __init__(self, model_dir: str = MODEL_DIR, data_path: str = 'Crop_recommendationV2.csv'):
        """
        Initializes the RecommendationSystem, attempting to load models or training them if not found.
        Args:
            model_dir (str): Directory where trained models and preprocessing objects are stored.
            data_path (str): Path to the raw crop recommendation CSV data.
        """
        self.model_dir = model_dir
        self.data_path = data_path
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.dl_ensemble_models: List[Model] = []
        self.rf_ensemble_models: List[RandomForestRegressor] = []
        self.gb_ensemble_models: List[GradientBoostingRegressor] = []
        self.feature_columns_for_prediction: List[str] = [] # These will be the features X was trained on
        self.target_indicators: List[str] = ['THI', 'NBR', 'WAI', 'PP', 'SFI']
        self.crop_ideal_indicator_ranges: Dict[str, Dict[str, tuple[float, float]]] = {}
        self.optimal_sensor_combinations: Dict[str, List[Dict[str, float]]] = {} # New attribute for optimal sensor combinations
        self.mean_training_features: Optional[Dict[str, float]] = None # Store means of non-controllable features
        self.controllable_sensors_config = {
            'N': {'step': 5.0, 'action_type': 'adjust_nutrient', 'nutrient_type': 'N', 'action_param_name': 'amount_mg', 'scale_factor': 5.0, 'min_val': 0.0, 'max_val': 200.0},
            'P': {'step': 5.0, 'action_type': 'adjust_nutrient', 'nutrient_type': 'P', 'action_param_name': 'amount_mg', 'scale_factor': 5.0, 'min_val': 0.0, 'max_val': 200.0},
            'K': {'step': 5.0, 'action_type': 'adjust_nutrient', 'nutrient_type': 'K', 'action_param_name': 'amount_mg', 'scale_factor': 5.0, 'min_val': 0.0, 'max_val': 200.0},
            'soil_moisture': {'step': 5.0, 'action_type': 'water_crop', 'action_param_name': 'amount_ml', 'scale_factor': 100.0, 'min_val': 0.0, 'max_val': 100.0},
            'sunlight_exposure': {'step': 5.0, 'action_type': 'adjust_lighting', 'action_param_name': 'intensity_percent', 'scale_factor': 1.0, 'min_val': 0.0, 'max_val': 140.0},
        }
        self._load_or_train_models()

    def _load_or_train_models(self) -> None:
        """
        Attempts to load trained models and preprocessing objects from `model_dir`.
        If any component is not found or an error occurs, it triggers the training process.
        """
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        pca_path = os.path.join(self.model_dir, 'pca.pkl')
        le_path = os.path.join(self.model_dir, 'label_encoder.pkl')
        crop_ranges_path = os.path.join(self.model_dir, 'crop_ideal_indicator_ranges.pkl')
        optimal_sensor_combinations_path = os.path.join(self.model_dir, 'optimal_sensor_combinations.pkl') # New path
        feature_columns_path = os.path.join(self.model_dir, 'feature_columns.pkl')
        mean_training_features_path = os.path.join(self.model_dir, 'mean_training_features.pkl') # New path for means

        models_loaded = False
        try:
            print("\nAttempting to load trained models and preprocessing objects...")
            self.scaler = joblib.load(scaler_path)
            self.pca = joblib.load(pca_path)
            self.label_encoder = joblib.load(le_path)

            for i in range(3): # Assuming 3 DL models
                self.dl_ensemble_models.append(load_model(os.path.join(self.model_dir, f'dl_model_{i}.keras')))
            
            for i in range(len(self.target_indicators)):
                self.rf_ensemble_models.append(joblib.load(os.path.join(self.model_dir, f'rf_model_{i}.pkl')))

            for i in range(len(self.target_indicators)):
                self.gb_ensemble_models.append(joblib.load(os.path.join(self.model_dir, f'gb_model_{i}.pkl')))
            
            self.crop_ideal_indicator_ranges = joblib.load(crop_ranges_path)
            self.optimal_sensor_combinations = joblib.load(optimal_sensor_combinations_path) # Load optimal sensor combinations
            self.feature_columns_for_prediction = joblib.load(feature_columns_path) # Load feature columns X was trained on
            self.mean_training_features = joblib.load(mean_training_features_path) # Load mean training features

            models_loaded = True
            print("Models and preprocessing objects loaded successfully. Skipping training.")

        except FileNotFoundError:
            print("Trained models not found. Proceeding with training...")
            models_loaded = False
        except Exception as e:
            print(f"Error loading models: {e}. Proceeding with training.")
            models_loaded = False

        if not models_loaded:
            self._train_and_save_models()

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs data cleaning, outlier handling, label encoding, and feature engineering.
        Args:
            df (pd.DataFrame): The raw input DataFrame.
        Returns:
            pd.DataFrame: The cleaned and feature-engineered DataFrame.
        """
        numeric_df = df.select_dtypes(include=np.number)
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
        df_cleaned = df[~outlier_mask].copy()

        self.label_encoder = LabelEncoder()
        df_cleaned['label_encoded'] = self.label_encoder.fit_transform(df_cleaned['label'])

        df_cleaned['THI'] = df_cleaned['temperature'] - (0.55 - 0.0055 * df_cleaned['humidity']) * (df_cleaned['temperature'] - 14.5)
        
        alpha_nbr = 1.5
        beta_nbr = 0.5
        df_cleaned['P_div_N'] = df_cleaned.apply(lambda row: row['P'] / row['N'] if row['N'] != 0 else 0, axis=1)
        df_cleaned['NBR'] = df_cleaned['N'] + alpha_nbr * df_cleaned['P_div_N'] + beta_nbr
        df_cleaned.drop('P_div_N', axis=1, inplace=True)

        Ra = 15
        T_mean = df_cleaned['temperature']
        T_max = df_cleaned['temperature'] + 5
        T_min = df_cleaned['temperature'] - 5
        temp_diff_sqrt = np.sqrt(np.maximum(0, T_max - T_min))
        df_cleaned['PET'] = 0.0023 * Ra * (T_mean + 17.8) * temp_diff_sqrt
        df_cleaned['WAI'] = df_cleaned['soil_moisture'] + df_cleaned['rainfall'] - df_cleaned['PET']

        df_cleaned['PP'] = (df_cleaned['sunlight_exposure'] * df_cleaned['co2_concentration'] * df_cleaned['temperature']) / 10000.0

        optimal_pH = 6.5
        gamma_sfi = 2
        pH_stress_factor = 1 - (np.abs(df_cleaned['ph'] - optimal_pH) / optimal_pH)**gamma_sfi
        df_cleaned['SFI'] = df_cleaned['organic_matter'] * df_cleaned['NBR'] * pH_stress_factor
        
        return df_cleaned

    def _apply_feature_engineering_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the same feature engineering steps as during training, but for a single prediction.
        Ensures consistency in feature creation.
        Args:
            df (pd.DataFrame): A DataFrame containing raw sensor readings and potentially other base features.
        Returns:
            pd.DataFrame: The DataFrame with engineered features.
        """
        # Ensure all necessary raw columns are present, fill with 0 if not
        for col in ['N', 'P', 'K', 'temperature', 'humidity', 'soil_moisture', 'rainfall', 'sunlight_exposure', 'co2_concentration', 'ph', 'organic_matter']:
            if col not in df.columns:
                df[col] = 0.0
        
        # Apply feature engineering steps
        df['THI'] = df['temperature'] - (0.55 - 0.0055 * df['humidity']) * (df['temperature'] - 14.5)
        
        alpha_nbr = 1.5
        beta_nbr = 0.5
        df['P_div_N'] = df.apply(lambda row: row['P'] / row['N'] if row['N'] != 0 else 0, axis=1)
        df['NBR'] = df['N'] + alpha_nbr * df['P_div_N'] + beta_nbr
        df.drop('P_div_N', axis=1, inplace=True, errors='ignore')

        Ra = 15
        T_mean = df['temperature']
        T_max = df['temperature'] + 5
        T_min = df['temperature'] - 5
        temp_diff_sqrt = np.sqrt(np.maximum(0, T_max - T_min))
        df['PET'] = 0.0023 * Ra * (T_mean + 17.8) * temp_diff_sqrt
        df['WAI'] = df['soil_moisture'] + df['rainfall'] - df['PET']
        df.drop('PET', axis=1, inplace=True, errors='ignore')

        df['PP'] = (df['sunlight_exposure'] * df['co2_concentration'] * df['temperature']) / 10000.0

        optimal_pH = 6.5
        gamma_sfi = 2
        pH_stress_factor = 1 - (np.abs(df['ph'] - optimal_pH) / optimal_pH)**gamma_sfi
        df['SFI'] = df['organic_matter'] * df['NBR'] * pH_stress_factor
        
        return df

    def _create_base_dl_model(self, input_dim: int, output_dim: int) -> Model:
        """
        Creates a sequential Deep Learning model for regression.
        Args:
            input_dim (int): The number of input features.
            output_dim (int): The number of output targets.
        Returns:
            Model: A compiled Keras Sequential model.
        """
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(output_dim, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
        return model

    def _train_and_save_models(self) -> None:
        """
        Trains all ensemble models (DL, RandomForest, GradientBoosting) and saves them
        along with preprocessing objects (scaler, PCA, label encoder) and crop ideal ranges.
        """
        df = pd.read_csv(self.data_path)
        df_cleaned = self._preprocess_data(df.copy()) # df_cleaned now has all features including label_encoded and engineered indicators

        # X contains all features used for training the indicator prediction models
        # This includes raw sensor data, other original features, and label_encoded
        X = df_cleaned.drop(['label'] + self.target_indicators, axis=1)
        y = df_cleaned[self.target_indicators]
        self.feature_columns_for_prediction = X.columns.tolist() # Save the list of features X was trained on
        
        # Calculate and store means for non-controllable features
        non_controllable_non_label_features = [
            f for f in self.feature_columns_for_prediction 
            if f not in self.controllable_sensors_config and f != 'label_encoded'
        ]
        self.mean_training_features = X[non_controllable_non_label_features].mean().to_dict()

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.pca = PCA(n_components=0.95)
        X_pca = self.pca.fit_transform(X_scaled)

        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

        all_model_test_predictions: List[np.ndarray] = []

        print("\nTraining Diverse Models for Ensemble...")

        num_dl_models = 3
        dl_test_predictions: List[np.ndarray] = []
        for i in range(num_dl_models):
            print(f"Training Deep Learning Model {i+1}/{num_dl_models}...")
            dl_model = self._create_base_dl_model(X_train.shape[1], len(self.target_indicators))
            dl_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
            
            loss, mae = dl_model.evaluate(X_test, y_test, verbose=0)
            print(f"DL Model {i+1} - Mean Squared Error: {loss:.4f}, Mean Absolute Error: {mae:.4f}")
            
            self.dl_ensemble_models.append(dl_model)
            dl_test_predictions.append(dl_model.predict(X_test, verbose=0))

        if dl_test_predictions:
            avg_dl_test_predictions = np.mean(dl_test_predictions, axis=0)
            all_model_test_predictions.append(avg_dl_test_predictions)

        print("\nTraining Random Forest Regressor (multi-output)...")
        rf_preds_test_individual: List[np.ndarray] = []
        for i, indicator in enumerate(self.target_indicators):
            print(f"  Training Random Forest for {indicator}...")
            rf_model_single = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_model_single.fit(X_train, y_train[indicator])
            rf_preds_test_individual.append(rf_model_single.predict(X_test))
            self.rf_ensemble_models.append(rf_model_single)

        rf_pred_test_combined = np.column_stack(rf_preds_test_individual)
        rf_loss = np.mean((rf_pred_test_combined - y_test.values)**2)
        rf_mae = np.mean(np.abs(rf_pred_test_combined - y_test.values))
        print(f"Random Forest Ensemble - Mean Squared Error: {rf_loss:.4f}, Mean Absolute Error: {rf_mae:.4f}")
        all_model_test_predictions.append(rf_pred_test_combined)

        print("\nTraining Gradient Boosting Regressor (multi-output)...")
        gb_preds_test_individual: List[np.ndarray] = []
        for i, indicator in enumerate(self.target_indicators):
            print(f"  Training Gradient Boosting for {indicator}...")
            gb_model_single = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gb_model_single.fit(X_train, y_train[indicator])
            gb_preds_test_individual.append(gb_model_single.predict(X_test))
            self.gb_ensemble_models.append(gb_model_single)

        gb_pred_test_combined = np.column_stack(gb_preds_test_individual)
        gb_loss = np.mean((gb_pred_test_combined - y_test.values)**2)
        gb_mae = np.mean(np.abs(gb_pred_test_combined - y_test.values))
        print(f"Gradient Boosting Ensemble - Mean Squared Error: {gb_loss:.4f}, Mean Absolute Error: {gb_mae:.4f}")
        all_model_test_predictions.append(gb_pred_test_combined)

        y_pred_overall_ensemble = np.mean(all_model_test_predictions, axis=0)
        overall_ensemble_loss = np.mean((y_pred_overall_ensemble - y_test.values)**2)
        overall_ensemble_mae = np.mean(np.abs(y_pred_overall_ensemble - y_test.values))

        print(f"\nOverall Ensemble Model Mean Squared Error (averaged predictions): {overall_ensemble_loss:.4f}")
        print(f"Overall Ensemble Model Mean Absolute Error (averaged predictions): {overall_ensemble_mae:.4f}")

        print("\nSaving trained models and preprocessing objects...")
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))
        joblib.dump(self.pca, os.path.join(self.model_dir, 'pca.pkl'))
        joblib.dump(self.label_encoder, os.path.join(self.model_dir, 'label_encoder.pkl'))

        for i, dl_model in enumerate(self.dl_ensemble_models):
            dl_model.save(os.path.join(self.model_dir, f'dl_model_{i}.keras'))

        for i, rf_model in enumerate(self.rf_ensemble_models):
            joblib.dump(rf_model, os.path.join(self.model_dir, f'rf_model_{i}.pkl'))

        for i, gb_model in enumerate(self.gb_ensemble_models):
            joblib.dump(gb_model, os.path.join(self.model_dir, f'gb_model_{i}.pkl'))
        
        joblib.dump(self.feature_columns_for_prediction, os.path.join(self.model_dir, 'feature_columns.pkl')) # Save engineered feature columns
        joblib.dump(self.mean_training_features, os.path.join(self.model_dir, 'mean_training_features.pkl')) # Save mean training features
        self._calculate_and_save_crop_ideal_ranges(df_cleaned)
        self._sample_optimal_sensor_combinations(df_cleaned) # Call the new sampling method
        print("Models and preprocessing objects saved successfully.")

    def _calculate_and_save_crop_ideal_ranges(self, df_cleaned: pd.DataFrame) -> None:
        """
        Calculates and saves crop-specific ideal indicator ranges based on mean +/- 1 standard deviation.
        Args:
            df_cleaned (pd.DataFrame): The cleaned DataFrame containing crop data and derived indicators.
        """
        crop_ideal_indicator_ranges: Dict[str, Dict[str, tuple[float, float]]] = {}
        for crop_label in df_cleaned['label'].unique():
            crop_data = df_cleaned[df_cleaned['label'] == crop_label]
            ranges: Dict[str, tuple[float, float]] = {}
            for indicator in self.target_indicators:
                mean_val = crop_data[indicator].mean()
                std_val = crop_data[indicator].std()
                min_val = mean_val - (2 * std_val) # Widen the range to 2 standard deviations
                max_val = mean_val + (2 * std_val) # Widen the range to 2 standard deviations
                ranges[indicator] = (min_val, max_val)
            crop_ideal_indicator_ranges[crop_label] = ranges
        self.crop_ideal_indicator_ranges = crop_ideal_indicator_ranges
        joblib.dump(self.crop_ideal_indicator_ranges, os.path.join(self.model_dir, 'crop_ideal_indicator_ranges.pkl'))
        print("\nCalculated and saved Crop-Specific Ideal Indicator Ranges (Mean +/- 1 Std Dev).")

    def _sample_optimal_sensor_combinations(self, df_cleaned: pd.DataFrame, num_samples_per_crop: int = 50) -> None:
        """
        Identifies and stores optimal combinations of controllable sensor readings for each crop
        by filtering existing data points that result in target indicators being within their ideal ranges.
        This replaces the inefficient random sampling with a more direct approach using the training data.
        Args:
            df_cleaned (pd.DataFrame): The cleaned DataFrame containing crop data and derived indicators.
            num_samples_per_crop (int): The maximum number of optimal combinations to store for each crop.
        """
        print(f"\nIdentifying optimal sensor combinations from training data for {len(df_cleaned['label'].unique())} crops (targeting up to {num_samples_per_crop} samples per crop)...")
        optimal_combinations: Dict[str, List[Dict[str, float]]] = {}

        controllable_sensor_names = list(self.controllable_sensors_config.keys())

        for crop_label_str in df_cleaned['label'].unique():
            print(f"  Processing crop: {crop_label_str}...")
            crop_optimal_sensors: List[Dict[str, float]] = []
            
            if crop_label_str not in self.crop_ideal_indicator_ranges:
                print(f"    No ideal indicator ranges for {crop_label_str}. Skipping.")
                continue

            ideal_ranges = self.crop_ideal_indicator_ranges[crop_label_str]
            
            # Filter data for the current crop
            crop_data = df_cleaned[df_cleaned['label'] == crop_label_str]

            # Iterate through each row of the crop's data
            for _, row in crop_data.iterrows():
                if len(crop_optimal_sensors) >= num_samples_per_crop:
                    break # Stop if we've collected enough samples for this crop

                is_optimal = True
                current_indicators = {indicator: row[indicator] for indicator in self.target_indicators}

                for indicator, value in current_indicators.items():
                    if indicator not in ideal_ranges:
                        is_optimal = False
                        break
                    ideal_min, ideal_max = ideal_ranges[indicator]
                    if not (ideal_min <= value <= ideal_max):
                        is_optimal = False
                        break
                
                if is_optimal:
                    # Extract controllable sensor values from this optimal row
                    controllable_sensor_values = {
                        sensor_name: row[sensor_name] 
                        for sensor_name in controllable_sensor_names 
                        if sensor_name in row
                    }
                    crop_optimal_sensors.append(controllable_sensor_values)
            
            optimal_combinations[crop_label_str] = crop_optimal_sensors
            print(f"    Found {len(crop_optimal_sensors)} optimal sensor combinations for {crop_label_str}.")

        self.optimal_sensor_combinations = optimal_combinations
        joblib.dump(self.optimal_sensor_combinations, os.path.join(self.model_dir, 'optimal_sensor_combinations.pkl'))
        print("Saved optimal sensor combinations.")

    def predict_indicators(self, input_data: Dict[str, float], crop_label: str) -> Dict[str, float]:
        """
        Predicts target indicators based on input sensor data and crop label using the ensemble model.
        Args:
            input_data (Dict[str, float]): A dictionary of current raw sensor readings (controllable sensors).
            crop_label (str): The type of crop for which to make predictions.
        Returns:
            Dict[str, float]: Predicted values for each target indicator.
        """
        if not all([self.scaler, self.pca, self.dl_ensemble_models, self.rf_ensemble_models, self.gb_ensemble_models, self.feature_columns_for_prediction, self.label_encoder]):
            raise RuntimeError("Recommendation system components not fully loaded or trained.")

        # Create a DataFrame that matches the features X was trained on (self.feature_columns_for_prediction)
        # This includes raw sensor data, other original features, and label_encoded
        
        # Start with a base DataFrame for the current prediction
        predict_df = pd.DataFrame([input_data])

        # Add label_encoded
        if 'label_encoded' in self.feature_columns_for_prediction:
            encoded_label = self.label_encoder.transform([crop_label])[0]
            predict_df['label_encoded'] = encoded_label

        # Add other original features that are in self.feature_columns_for_prediction but not in input_data
        # For these, we'll use a representative value (e.g., mean from training data)
        # This requires access to the original training data or pre-calculated means.
        # For simplicity, let's load the original data and calculate means if not already done.
        # Add other original features that are in self.feature_columns_for_prediction but not in input_data
        # For these, we'll use the pre-calculated mean from training data.
        if self.mean_training_features is None:
            raise RuntimeError("Mean training features not loaded or trained.")
        
        for col in self.feature_columns_for_prediction:
            if col not in predict_df.columns:
                if col in self.mean_training_features:
                    predict_df[col] = self.mean_training_features[col] # Use pre-calculated mean
                else:
                    predict_df[col] = 0.0 # Fallback if feature not found even in stored means

        # Ensure column order matches training
        predict_df = predict_df[self.feature_columns_for_prediction]

        input_scaled = self.scaler.transform(predict_df)
        input_pca = self.pca.transform(input_scaled)

        all_individual_predictions: List[np.ndarray] = []

        dl_individual_predictions = [model.predict(input_pca, verbose=0)[0] for model in self.dl_ensemble_models]
        if dl_individual_predictions:
            all_individual_predictions.append(np.mean(dl_individual_predictions, axis=0))

        rf_scalar_predictions = [model.predict(input_pca)[0] for model in self.rf_ensemble_models]
        if rf_scalar_predictions:
            all_individual_predictions.append(np.array(rf_scalar_predictions))

        gb_scalar_predictions = [model.predict(input_pca)[0] for model in self.gb_ensemble_models]
        if gb_scalar_predictions:
            all_individual_predictions.append(np.array(gb_scalar_predictions))
        
        ensembled_predictions = np.mean(all_individual_predictions, axis=0)
        
        return dict(zip(self.target_indicators, ensembled_predictions))

    def _calculate_indicator_gradients(self, current_sensor_data: Dict[str, float], 
                                        predicted_indicators_dict: Dict[str, float],
                                        crop_label: str) -> Dict[str, Dict[str, float]]:
        """
        Calculates the approximate gradient (impact per unit change) of each controllable sensor
        on each target indicator using perturbation.
        Args:
            current_sensor_data (Dict[str, float]): The current raw sensor readings.
            predicted_indicators_dict (Dict[str, float]): Predicted values for each target indicator.
            crop_label (str): The type of crop being grown.
        Returns:
            Dict[str, Dict[str, float]]: A dictionary where keys are sensor names, and values are
                                         dictionaries of indicator impacts (gradients).
        """
        gradients: Dict[str, Dict[str, float]] = {sensor: {} for sensor in self.controllable_sensors_config.keys()}

        for sensor_name, config in self.controllable_sensors_config.items():
            if sensor_name not in current_sensor_data:
                continue

            original_sensor_value = current_sensor_data[sensor_name]
            perturbation_step = config['step']

            # Perturb upwards
            perturbed_data_increase = current_sensor_data.copy()
            perturbed_data_increase[sensor_name] = min(original_sensor_value + perturbation_step, config['max_val'])
            predicted_after_increase = self.predict_indicators(perturbed_data_increase, crop_label)

            # Perturb downwards
            perturbed_data_decrease = current_sensor_data.copy()
            perturbed_data_decrease[sensor_name] = max(original_sensor_value - perturbation_step, config['min_val'])
            predicted_after_decrease = self.predict_indicators(perturbed_data_decrease, crop_label)

            for indicator in self.target_indicators:
                impact_increase = predicted_after_increase.get(indicator, predicted_indicators_dict.get(indicator, 0.0)) - predicted_indicators_dict.get(indicator, 0.0)
                impact_decrease = predicted_after_decrease.get(indicator, predicted_indicators_dict.get(indicator, 0.0)) - predicted_indicators_dict.get(indicator, 0.0)
                
                # Approximate gradient as average of forward and backward perturbation impacts
                # Or, if only one direction is possible due to bounds, use that.
                if original_sensor_value + perturbation_step > config['max_val'] and original_sensor_value - perturbation_step < config['min_val']:
                    gradient = 0.0 # Cannot perturb in either direction
                elif original_sensor_value + perturbation_step > config['max_val']:
                    gradient = impact_decrease / (-perturbation_step) if perturbation_step != 0 else 0.0
                elif original_sensor_value - perturbation_step < config['min_val']:
                    gradient = impact_increase / perturbation_step if perturbation_step != 0 else 0.0
                else:
                    gradient = (impact_increase - impact_decrease) / (2 * perturbation_step) if perturbation_step != 0 else 0.0
                
                gradients[sensor_name][indicator] = gradient
        return gradients

    def _calculate_optimal_sensor_adjustments(self, current_sensor_data: Dict[str, float], 
                                              predicted_indicators_dict: Dict[str, float], # Keep this for fallback
                                              crop_label: str) -> List[Dict[str, Any]]:
        """
        Calculates optimal sensor adjustments to steer current sensor readings towards
        a sampled optimal sensor combination for the given crop.
        Args:
            current_sensor_data (Dict[str, float]): The current raw sensor readings.
            predicted_indicators_dict (Dict[str, float]): Predicted values for each target indicator (for fallback).
            crop_label (str): The type of crop being grown.
        Returns:
            List[Dict[str, Any]]: A list of structured recommendations for sensor adjustments.
        """
        recommendations: List[Dict[str, Any]] = []

        if crop_label not in self.optimal_sensor_combinations or not self.optimal_sensor_combinations[crop_label]:
            recommendations.append({"action": "notify_user", "message": f"No optimal sensor combinations found for crop: {crop_label}. Falling back to indicator-based adjustments."})
            # Fallback to the previous logic if no optimal sensor combinations are found
            # This requires the predicted_indicators_dict, which is why it's kept as a parameter.
            ideal_ranges = self.crop_ideal_indicator_ranges[crop_label]
            gradients = self._calculate_indicator_gradients(current_sensor_data, predicted_indicators_dict, crop_label)

            indicator_targets: Dict[str, float] = {}
            for indicator, predicted_value in predicted_indicators_dict.items():
                if indicator not in ideal_ranges:
                    continue
                ideal_min, ideal_max = ideal_ranges[indicator]
                if predicted_value < ideal_min:
                    indicator_targets[indicator] = ideal_min - predicted_value
                elif predicted_value > ideal_max:
                    indicator_targets[indicator] = ideal_max - predicted_value
            
            if not indicator_targets:
                recommendations.append({"action": "notify_user", "message": "All indicators are within ideal ranges. No adjustments needed."})
                return recommendations

            sensor_adjustments: Dict[str, float] = {sensor: 0.0 for sensor in self.controllable_sensors_config.keys()}
            learning_rate = 0.5

            for sensor_name, sensor_config in self.controllable_sensors_config.items():
                if sensor_name not in current_sensor_data:
                    continue
                total_adjustment_for_sensor = 0.0
                for indicator, target_change in indicator_targets.items():
                    gradient = gradients.get(sensor_name, {}).get(indicator, 0.0)
                    if abs(gradient) > 1e-5:
                        total_adjustment_for_sensor += (target_change / gradient) * abs(target_change)
                
                if any(abs(indicator_targets[ind]) > 0 for ind in indicator_targets):
                    sensor_adjustments[sensor_name] = learning_rate * (total_adjustment_for_sensor / sum(abs(tc) for tc in indicator_targets.values()))
                else:
                    sensor_adjustments[sensor_name] = 0.0

                original_sensor_value = current_sensor_data[sensor_name]
                proposed_new_value = original_sensor_value + sensor_adjustments[sensor_name]
                clamped_value = max(sensor_config['min_val'], min(proposed_new_value, sensor_config['max_val']))
                sensor_adjustments[sensor_name] = clamped_value - original_sensor_value
            
            # Generate structured recommendations from calculated adjustments (fallback logic)
            for sensor_name, raw_sensor_change in sensor_adjustments.items():
                if abs(raw_sensor_change) < 0.01:
                    continue
                sensor_config = self.controllable_sensors_config[sensor_name]
                action_amount = 0.0
                action_message_suffix = ""
                if sensor_config['action_type'] == 'water_crop':
                    action_amount = max(0, min(1000, raw_sensor_change * sensor_config.get('scale_factor', 1.0)))
                    action_message_suffix = f" ({round(action_amount, 2)} ml)"
                elif sensor_config['action_type'] == 'adjust_nutrient':
                    action_amount = max(0, min(200, raw_sensor_change * sensor_config.get('scale_factor', 1.0)))
                    action_message_suffix = f" ({round(action_amount, 2)} mg)"
                elif sensor_config['action_type'] == 'adjust_lighting':
                    current_sunlight_exposure = current_sensor_data[sensor_name]
                    target_sunlight_exposure = current_sunlight_exposure + raw_sensor_change
                    sunlight_min = sensor_config['min_val']
                    sunlight_max = sensor_config['max_val']
                    target_sunlight_exposure = max(sunlight_min, min(sunlight_max, target_sunlight_exposure))
                    intensity_percent = ((target_sunlight_exposure - sunlight_min) / (sunlight_max - sunlight_min)) * 100.0
                    action_amount = max(0, min(100, intensity_percent))
                    action_message_suffix = f" ({round(action_amount, 2)}%)"
                action_dict = {
                    "action": sensor_config['action_type'],
                    sensor_config['action_param_name']: round(action_amount, 2)
                }
                if 'nutrient_type' in sensor_config:
                    action_dict['nutrient_type'] = sensor_config['nutrient_type']
                recommendations.append(action_dict)
                recommendations.append({"action": "notify_user", "message": f"Recommended {sensor_config['action_type']}{action_message_suffix} for {sensor_name} (change: {raw_sensor_change:.2f})."})
            
            if not recommendations:
                recommendations.append({"action": "notify_user", "message": "No significant sensor adjustments calculated (fallback)."})
            return recommendations


        # 1. Find the closest optimal combination
        current_controllable_sensors = {s: current_sensor_data.get(s, 0.0) for s in self.controllable_sensors_config.keys()}
        
        min_distance = float('inf')
        target_optimal_sensors: Optional[Dict[str, float]] = None

        for optimal_combo in self.optimal_sensor_combinations[crop_label]:
            distance = 0.0
            for sensor_name in self.controllable_sensors_config.keys():
                # Ensure sensor exists in both current data and optimal combo
                if sensor_name in current_controllable_sensors and sensor_name in optimal_combo:
                    distance += (current_controllable_sensors[sensor_name] - optimal_combo[sensor_name])**2
            distance = np.sqrt(distance)

            if distance < min_distance:
                min_distance = distance
                target_optimal_sensors = optimal_combo
        
        if not target_optimal_sensors:
            recommendations.append({"action": "notify_user", "message": f"Could not determine a target optimal sensor combination for crop: {crop_label}."})
            return recommendations

        recommendations.append({"action": "notify_user", "message": f"Targeting closest optimal sensor combination (distance: {min_distance:.2f})."})
        recommendations.append({"action": "notify_user", "message": f"Target Optimal Sensors: {target_optimal_sensors}"})

        # 2. Calculate required sensor changes
        sensor_adjustments: Dict[str, float] = {}
        for sensor_name, target_value in target_optimal_sensors.items():
            if sensor_name in current_sensor_data:
                current_value = current_sensor_data[sensor_name]
                raw_change = target_value - current_value
                
                # Apply bounds to the raw sensor adjustment
                sensor_config = self.controllable_sensors_config[sensor_name]
                proposed_new_value = current_value + raw_change
                clamped_value = max(sensor_config['min_val'], min(proposed_new_value, sensor_config['max_val']))
                
                sensor_adjustments[sensor_name] = clamped_value - current_value # Store the actual change after clamping
            else:
                sensor_adjustments[sensor_name] = 0.0 # Sensor not in current data, no change

        # 3. Generate structured recommendations from calculated adjustments
        for sensor_name, raw_sensor_change in sensor_adjustments.items():
            if abs(raw_sensor_change) < 0.01: # Only recommend significant changes
                continue

            sensor_config = self.controllable_sensors_config[sensor_name]
            action_amount = 0.0
            action_message_suffix = ""

            if sensor_config['action_type'] == 'water_crop':
                action_amount = max(0, min(1000, raw_sensor_change * sensor_config.get('scale_factor', 1.0))) # Amount in ml
                action_message_suffix = f" ({round(action_amount, 2)} ml)"
            elif sensor_config['action_type'] == 'adjust_nutrient':
                action_amount = max(0, min(200, raw_sensor_change * sensor_config.get('scale_factor', 1.0))) # Amount in mg
                action_message_suffix = f" ({round(action_amount, 2)} mg)"
            elif sensor_config['action_type'] == 'adjust_lighting':
                # Convert target sunlight_exposure to intensity_percent (0-100)
                current_sunlight_exposure = current_sensor_data[sensor_name]
                target_sunlight_exposure = current_sunlight_exposure + raw_sensor_change
                
                sunlight_min = sensor_config['min_val']
                sunlight_max = sensor_config['max_val']
                
                # Cap target_sensor_value within sensor's min/max
                target_sunlight_exposure = max(sunlight_min, min(sunlight_max, target_sunlight_exposure))
                
                # Convert to percentage
                intensity_percent = ((target_sunlight_exposure - sunlight_min) / (sunlight_max - sunlight_min)) * 100.0
                action_amount = max(0, min(100, intensity_percent)) # Cap percentage between 0-100
                action_message_suffix = f" ({round(action_amount, 2)}%)"
            
            action_dict = {
                "action": sensor_config['action_type'],
                sensor_config['action_param_name']: round(action_amount, 2)
            }
            if 'nutrient_type' in sensor_config:
                action_dict['nutrient_type'] = sensor_config['nutrient_type']
            recommendations.append(action_dict)
            recommendations.append({"action": "notify_user", "message": f"Recommended {sensor_config['action_type']}{action_message_suffix} for {sensor_name} (change: {raw_sensor_change:.2f})."})
        
        if not recommendations:
            recommendations.append({"action": "notify_user", "message": "No significant sensor adjustments calculated."})

        return recommendations

    def recommend_sensor_changes_from_indicators(self, predicted_indicators_dict: Dict[str, float], crop_label: str, current_sensor_data: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Generates structured recommendations for sensor changes based on predicted indicators
        and crop-specific ideal ranges, now primarily using sampled optimal sensor combinations.
        Args:
            predicted_indicators_dict (Dict[str, float]): Predicted values for each target indicator (still passed for potential fallback/logging).
            crop_label (str): The type of crop being grown.
            current_sensor_data (Dict[str, float]): The current raw sensor readings (needed for adjustment calculation).
        Returns:
            List[Dict[str, Any]]: A list of structured recommendations. Each dict contains 'action', 'target', 'value', etc.
        """
        # Use the new method that targets optimal sensor combinations
        intelligent_recommendations = self._calculate_optimal_sensor_adjustments(
            current_sensor_data, predicted_indicators_dict, crop_label
        )
        
        return intelligent_recommendations

if __name__ == "__main__":
    print("Initializing Recommendation System for training/loading...")
    rec_system = RecommendationSystem()
    print("Recommendation System ready.")
    
    df = pd.read_csv('Crop_recommendationV2.csv')
    df_cleaned = rec_system._preprocess_data(df.copy())
    
    # Sample raw sensor data for prediction
    sample_raw_sensor_data = {
        'N': 90.0, 'P': 42.0, 'K': 43.0, 'temperature': 20.879744, 'humidity': 82.002744,
        'ph': 6.502985, 'rainfall': 202.935536, 'soil_moisture': 60.0, 'sunlight_exposure': 100.0,
        'co2_concentration': 400.0, 'organic_matter': 1.5
    }
    sample_crop = 'rice' # Example crop

    predicted_indicators = rec_system.predict_indicators(sample_raw_sensor_data, sample_crop)
    print(f"\nSample Input Raw Sensors: {sample_raw_sensor_data}")
    print(f"Predicted Indicators (Ensemble) for {sample_crop}: {predicted_indicators}")

    recommendations = rec_system.recommend_sensor_changes_from_indicators(predicted_indicators, sample_crop, sample_raw_sensor_data)
    print(f"\nRecommendations for {sample_crop}:")
    for rec in recommendations:
        print(f"- {rec}")
