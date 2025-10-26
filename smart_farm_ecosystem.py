import time
import random
import pandas as pd
import numpy as np
from threading import Thread, Event, Lock
from collections import deque

# Placeholder for the recommendation system components
# These will be loaded from smart_farm_recommendation.py
scaler = None
pca = None
dl_ensemble_models = []
rf_ensemble_models = []
gb_ensemble_models = []
feature_columns_for_prediction = []
target_indicators = []
crop_ideal_indicator_ranges = {}
predict_indicators_ensemble_func = None
recommend_sensor_changes_from_indicators_func = None

# --- 1. Sensor Simulation Module ---
class Sensor:
    def __init__(self, name, unit, min_val, max_val, current_val=None, drift_rate=0.1, noise_level=0.5):
        self.name = name
        self.unit = unit
        self.min_val = min_val
        self.max_val = max_val
        self._current_val = current_val if current_val is not None else random.uniform(min_val, max_val)
        self.drift_rate = drift_rate
        self.noise_level = noise_level
        self.lock = Lock()

    def _simulate_drift(self):
        drift = random.uniform(-self.drift_rate, self.drift_rate)
        self._current_val += drift
        self._current_val = max(self.min_val, min(self.max_val, self._current_val))

    def _add_noise(self):
        noise = random.uniform(-self.noise_level, self.noise_level)
        self._current_val += noise
        self._current_val = max(self.min_val, min(self.max_val, self._current_val))

    def get_value(self):
        with self.lock:
            self._simulate_drift()
            self._add_noise()
            return round(self._current_val, 2)

    def set_value(self, new_val):
        with self.lock:
            self._current_val = max(self.min_val, min(self.max_val, new_val))
            print(f"Manual override: {self.name} set to {self._current_val:.2f} {self.unit}")

class SensorGroup:
    def __init__(self, group_name, sensors_config):
        self.group_name = group_name
        self.sensors = {s['name']: Sensor(s['name'], s['unit'], s['min'], s['max'], s.get('initial_val')) for s in sensors_config}
        self.lock = Lock()

    def get_all_sensor_data(self):
        with self.lock:
            return {name: sensor.get_value() for name, sensor in self.sensors.items()}

    def get_sensor(self, sensor_name):
        return self.sensors.get(sensor_name)

# Define sensor configurations for a typical farm plot
FARM_SENSORS_CONFIG = [
    {'name': 'N', 'unit': 'ppm', 'min': 0, 'max': 140, 'initial_val': 70},
    {'name': 'P', 'unit': 'ppm', 'min': 0, 'max': 145, 'initial_val': 60},
    {'name': 'K', 'unit': 'ppm', 'min': 0, 'max': 205, 'initial_val': 90},
    {'name': 'temperature', 'unit': '°C', 'min': 0, 'max': 45, 'initial_val': 25},
    {'name': 'humidity', 'unit': '%', 'min': 0, 'max': 100, 'initial_val': 70},
    {'name': 'ph', 'unit': '', 'min': 0, 'max': 14, 'initial_val': 6.5},
    {'name': 'rainfall', 'unit': 'mm', 'min': 0, 'max': 300, 'initial_val': 100},
    {'name': 'soil_moisture', 'unit': '%', 'min': 0, 'max': 100, 'initial_val': 50},
    {'name': 'sunlight_exposure', 'unit': 'lux', 'min': 0, 'max': 100000, 'initial_val': 50000},
    {'name': 'co2_concentration', 'unit': 'ppm', 'min': 300, 'max': 1000, 'initial_val': 400},
    {'name': 'organic_matter', 'unit': '%', 'min': 0, 'max': 10, 'initial_val': 3.5},
]

# --- 2. Control Device Module ---
class ControlDevice:
    def __init__(self, name, action_func):
        self.name = name
        self.action_func = action_func
        self.status = "idle"
        self.lock = Lock()

    def perform_action(self, *args, **kwargs):
        with self.lock:
            self.status = "active"
            print(f"[{self.name}] Performing action...")
            result = self.action_func(*args, **kwargs)
            self.status = "idle"
            return result

    def get_status(self):
        with self.lock:
            return self.status

# Define specific actions
def water_crop_action(plot_id, amount_ml):
    print(f"[ACTION] Plot {plot_id}: Watering crop with {amount_ml} ml of water.")
    # In a real system, this would interface with an irrigation system
    return f"Watered {plot_id} with {amount_ml} ml"

def notify_user_action(plot_id, message):
    print(f"[ACTION] Plot {plot_id}: User Notification: {message}")
    # In a real system, this would send an email, SMS, or app notification
    return f"Notified user for {plot_id}: {message}"

def adjust_lighting_action(plot_id, intensity_percent):
    print(f"[ACTION] Plot {plot_id}: Adjusting lighting to {intensity_percent}%.")
    # In a real system, this would control grow lights
    return f"Adjusted lighting for {plot_id} to {intensity_percent}%"

def adjust_nutrient_action(plot_id, nutrient_type, amount_mg):
    print(f"[ACTION] Plot {plot_id}: Adjusting {nutrient_type} by {amount_mg} mg.")
    # In a real system, this would control nutrient delivery systems
    return f"Adjusted {nutrient_type} for {plot_id} by {amount_mg} mg"

# --- 3. Crop Management Module ---
class Crop:
    def __init__(self, plot_id, crop_type, current_growth_stage="seedling"):
        self.plot_id = plot_id
        self.crop_type = crop_type
        self.current_growth_stage = current_growth_stage
        self.parameters = {
            'target_yield': 100, # Example parameter
            'disease_risk': 0.1, # Example parameter
            'pest_risk': 0.05,   # Example parameter
        }
        self.lock = Lock()

    def get_parameters(self):
        with self.lock:
            return self.parameters.copy()

    def update_parameter(self, param_name, value):
        with self.lock:
            if param_name in self.parameters:
                self.parameters[param_name] = value
                print(f"Crop {self.crop_type} (Plot {self.plot_id}): Parameter '{param_name}' updated to {value}")
                return True
            print(f"Crop {self.crop_type} (Plot {self.plot_id}): Parameter '{param_name}' not found.")
            return False

# --- 4. Orchestrator Module ---
class Orchestrator:
    def __init__(self, plot_id, crop_type, sensor_group, plot_control_devices, update_interval=5):
        self.plot_id = plot_id
        self.crop = Crop(plot_id, crop_type)
        self.sensor_group = sensor_group
        self.control_devices = plot_control_devices # Dict of ControlDevice objects
        self.update_interval = update_interval
        self._running = False
        self._thread = None
        self._stop_event = Event()
        self.data_history = deque(maxlen=100) # Store last 100 sensor readings
        self.recommendation_history = deque(maxlen=20) # Store last 20 recommendations

    def _run_orchestration_loop(self):
        global scaler, pca, dl_ensemble_models, rf_ensemble_models, gb_ensemble_models, \
               feature_columns_for_prediction, target_indicators, crop_ideal_indicator_ranges, \
               predict_indicators_ensemble_func, recommend_sensor_changes_from_indicators_func

        if not all([scaler, pca, dl_ensemble_models, rf_ensemble_models, gb_ensemble_models,
                    feature_columns_for_prediction, target_indicators, crop_ideal_indicator_ranges,
                    predict_indicators_ensemble_func, recommend_sensor_changes_from_indicators_func]):
            print(f"Orchestrator for Plot {self.plot_id}: Recommendation system not fully loaded. Skipping loop.")
            return

        while not self._stop_event.is_set():
            print(f"\n--- Orchestrator for Plot {self.plot_id} ({self.crop.crop_type}) ---")
            sensor_data = self.sensor_group.get_all_sensor_data()
            self.data_history.append((time.time(), sensor_data))
            print(f"Current Sensor Data: {sensor_data}")

            # Prepare data for prediction
            input_for_prediction = {k: sensor_data.get(k, 0) for k in feature_columns_for_prediction}
            
            # Predict indicators using the ensemble model
            predicted_indicators = predict_indicators_ensemble_func(
                input_for_prediction, scaler, pca, dl_ensemble_models, rf_ensemble_models, gb_ensemble_models,
                feature_columns_for_prediction, target_indicators
            )
            print(f"Predicted Indicators: {predicted_indicators}")

            # Get recommendations based on predicted indicators and crop type
            recommendations = recommend_sensor_changes_from_indicators_func(
                predicted_indicators, self.crop.crop_type, crop_ideal_indicator_ranges
            )
            self.recommendation_history.append((time.time(), recommendations))
            print("Recommendations:")
            for rec in recommendations:
                print(f"- {rec}")
                # Trigger actions based on recommendations
                self._trigger_action_from_recommendation(rec)

            time.sleep(self.update_interval)

    def _trigger_action_from_recommendation(self, recommendation_text):
        # Simple rule-based action triggering for demonstration
        if "increase conditions affecting WAI" in recommendation_text:
            self.control_devices['water_pump'].perform_action(self.plot_id, 500) # Water 500ml
        elif "decrease conditions affecting WAI" in recommendation_text:
            self.control_devices['notify_user'].perform_action(self.plot_id, "Soil moisture is too high, consider reducing irrigation.")
        elif "increase conditions affecting NBR" in recommendation_text:
            self.control_devices['nutrient_dispenser'].perform_action(self.plot_id, 'NPK', 100)
        elif "decrease conditions affecting NBR" in recommendation_text:
            self.control_devices['notify_user'].perform_action(self.plot_id, "Nutrient levels are too high, consider flushing or reducing fertilizer.")
        elif "increase conditions affecting PP" in recommendation_text:
            self.control_devices['lighting_system'].perform_action(self.plot_id, 80) # Increase light to 80%
        elif "decrease conditions affecting PP" in recommendation_text:
            self.control_devices['lighting_system'].perform_action(self.plot_id, 40) # Decrease light to 40%
        elif "increase conditions affecting THI" in recommendation_text:
            self.control_devices['notify_user'].perform_action(self.plot_id, "Temperature/Humidity Index is low, consider increasing temperature or humidity.")
        elif "decrease conditions affecting THI" in recommendation_text:
            self.control_devices['notify_user'].perform_action(self.plot_id, "Temperature/Humidity Index is high, consider decreasing temperature or humidity.")
        elif "increase conditions affecting SFI" in recommendation_text:
            self.control_devices['notify_user'].perform_action(self.plot_id, "Soil Fertility Index is low, consider adding organic matter or compost.")
        elif "decrease conditions affecting SFI" in recommendation_text:
            self.control_devices['notify_user'].perform_action(self.plot_id, "Soil Fertility Index is high, consider soil aeration or dilution.")


    def start(self):
        if not self._running:
            self._running = True
            self._stop_event.clear()
            self._thread = Thread(target=self._run_orchestration_loop)
            self._thread.daemon = True
            self._thread.start()
            print(f"Orchestrator for Plot {self.plot_id} started.")

    def stop(self):
        if self._running:
            self._running = False
            self._stop_event.set()
            if self._thread:
                self._thread.join()
            print(f"Orchestrator for Plot {self.plot_id} stopped.")

# --- Main Simulation Setup ---
class SmartFarmEcosystem:
    def __init__(self):
        self.sensor_groups = {} # plot_id -> SensorGroup
        self.orchestrators = {} # plot_id -> Orchestrator
        self.lock = Lock()

    def add_farm_plot(self, plot_id, crop_type, plot_control_devices, sensors_config=FARM_SENSORS_CONFIG, update_interval=5):
        with self.lock:
            if plot_id in self.orchestrators:
                print(f"Plot {plot_id} already exists.")
                return

            sensor_group = SensorGroup(f"Plot_{plot_id}_Sensors", sensors_config)
            orchestrator = Orchestrator(plot_id, crop_type, sensor_group, plot_control_devices, update_interval)
            
            self.sensor_groups[plot_id] = sensor_group
            self.orchestrators[plot_id] = orchestrator
            print(f"Added Farm Plot {plot_id} with {crop_type}.")
            return orchestrator

    def start_all_orchestrators(self):
        with self.lock:
            for orchestrator in self.orchestrators.values():
                orchestrator.start()

    def stop_all_orchestrators(self):
        with self.lock:
            for orchestrator in self.orchestrators.values():
                orchestrator.stop()

    def get_plot_info(self, plot_id):
        with self.lock:
            orchestrator = self.orchestrators.get(plot_id)
            if orchestrator:
                sensor_data = orchestrator.sensor_group.get_all_sensor_data()
                crop_params = orchestrator.crop.get_parameters()
                return {
                    'plot_id': plot_id,
                    'crop_type': orchestrator.crop.crop_type,
                    'sensor_data': sensor_data,
                    'crop_parameters': crop_params,
                    'orchestrator_status': "Running" if orchestrator._running else "Stopped",
                    'control_device_statuses': {name: dev.get_status() for name, dev in orchestrator.control_devices.items()},
                    'data_history_count': len(orchestrator.data_history),
                    'recommendation_history_count': len(orchestrator.recommendation_history)
                }
            return None

    def get_sensor_for_plot(self, plot_id, sensor_name):
        with self.lock:
            sensor_group = self.sensor_groups.get(plot_id)
            if sensor_group:
                return sensor_group.get_sensor(sensor_name)
            return None

    def update_crop_parameter(self, plot_id, param_name, value):
        with self.lock:
            orchestrator = self.orchestrators.get(plot_id)
            if orchestrator:
                return orchestrator.crop.update_parameter(param_name, value)
            print(f"Plot {plot_id} not found.")
            return False

# --- Utility to load recommendation system components ---
def load_recommendation_system_components(recommendation_module):
    global scaler, pca, dl_ensemble_models, rf_ensemble_models, gb_ensemble_models, \
           feature_columns_for_prediction, target_indicators, crop_ideal_indicator_ranges, \
           predict_indicators_ensemble_func, recommend_sensor_changes_from_indicators_func

    scaler = recommendation_module.scaler
    pca = recommendation_module.pca
    dl_ensemble_models = recommendation_module.dl_ensemble_models
    rf_ensemble_models = recommendation_module.rf_ensemble_models
    gb_ensemble_models = recommendation_module.gb_ensemble_models
    feature_columns_for_prediction = recommendation_module.feature_columns_for_prediction
    target_indicators = recommendation_module.target_indicators
    crop_ideal_indicator_ranges = recommendation_module.crop_ideal_indicator_ranges
    predict_indicators_ensemble_func = recommendation_module.predict_indicators_ensemble
    recommend_sensor_changes_from_indicators_func = recommendation_module.recommend_sensor_changes_from_indicators
    print("Recommendation system components loaded successfully.")

# Example Usage (to be run in a separate script or interactive session)
if __name__ == "__main__":
    # This part would typically be in a separate main script or interactive session
    # to avoid circular imports and manage the simulation lifecycle.
    print("This script defines the Smart Farm Ecosystem components.")
    print("To run a simulation, you would import these components and orchestrate them.")
    print("Example: ")
    print("  import smart_farm_ecosystem")
    print("  import smart_farm_recommendation as rec_sys")
    print("  smart_farm_ecosystem.load_recommendation_system_components(rec_sys)")
    print("  farm = smart_farm_ecosystem.SmartFarmEcosystem()")
    print("  farm.add_farm_plot('plot1', 'rice')")
    print("  farm.add_farm_plot('plot2', 'maize')")
    print("  farm.start_all_orchestrators()")
    print("  # To interact, e.g., change a sensor value:")
    print("  # sensor_n_plot1 = farm.get_sensor_for_plot('plot1', 'N')")
    print("  # if sensor_n_plot1: sensor_n_plot1.set_value(120)")
    print("  # To update a crop parameter:")
    print("  # farm.update_crop_parameter('plot1', 'target_yield', 150)")
    print("  # time.sleep(60) # Let it run for a while")
    print("  # farm.stop_all_orchestrators()")
