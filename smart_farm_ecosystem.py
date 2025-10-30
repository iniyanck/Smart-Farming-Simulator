import time
import random
import pandas as pd
import numpy as np
from threading import Thread, Event, Lock
from collections import deque
from typing import Dict, List, Any, Optional, Union

from smart_farm_recommendation import RecommendationSystem
from smart_farm_logs import farm_logger

# --- 1. Sensor Simulation Module ---
class Sensor:
    """
    Simulates a single sensor with drift and noise.
    """
    def __init__(self, name: str, unit: str, min_val: float, max_val: float, 
                 current_val: Optional[float] = None, drift_rate: float = 0.01, noise_level: float = 0.02):
        self.name = name
        self.unit = unit
        self.min_val = min_val
        self.max_val = max_val
        self._current_val = current_val if current_val is not None else random.uniform(min_val, max_val)
        self.drift_rate = drift_rate
        self.noise_level = noise_level
        self.lock = Lock()
        self.drying_rate = 0.5 if self.name == 'soil_moisture' else 0.0 # Specific drying rate for soil moisture
        self._manual_override = False

    def _simulate_drift(self) -> None:
        """Applies a random drift to the sensor's current value and simulates drying for soil moisture."""
        drift = random.uniform(-self.drift_rate, self.drift_rate)
        self._current_val += drift
        
        if self.name == 'soil_moisture':
            self._current_val -= self.drying_rate # Simulate water drying up
            
        self._current_val = max(self.min_val, min(self.max_val, self._current_val))

    def _add_noise(self) -> None:
        """Adds random noise to the sensor's current value."""
        noise = random.uniform(-self.noise_level, self.noise_level)
        self._current_val += noise
        self._current_val = max(self.min_val, min(self.max_val, self._current_val))

    def get_value(self) -> float:
        """
        Retrieves the current sensor value after simulating drift and noise.
        Returns:
            float: The current (simulated) value of the sensor.
        """
        with self.lock:
            if self._manual_override:
                self._manual_override = False
                return round(self._current_val, 2)
            self._simulate_drift()
            self._add_noise()
            return round(self._current_val, 2)

    def set_value(self, new_val: float) -> None:
        """
        Manually sets a new value for the sensor, clamping it within min/max bounds.
        Args:
            new_val (float): The new value to set for the sensor.
        """
        with self.lock:
            self._current_val = max(self.min_val, min(self.max_val, new_val))
            self._manual_override = True
            farm_logger.info(f"Manual override: {self.name} set to {self._current_val:.2f} {self.unit}")

class SensorGroup:
    """
    Manages a collection of Sensor objects for a specific farm plot.
    """
    def __init__(self, group_name: str, sensors_config: List[Dict[str, Any]]):
        self.group_name = group_name
        self.sensors = {s['name']: Sensor(s['name'], s['unit'], s['min'], s['max'], s.get('initial_val')) for s in sensors_config}
        self.lock = Lock()

    def get_all_sensor_data(self) -> Dict[str, float]:
        """
        Retrieves the current values from all sensors in the group.
        Returns:
            Dict[str, float]: A dictionary where keys are sensor names and values are their readings.
        """
        with self.lock:
            return {name: sensor.get_value() for name, sensor in self.sensors.items()}

    def get_sensor(self, sensor_name: str) -> Optional[Sensor]:
        """
        Retrieves a specific sensor by its name.
        Args:
            sensor_name (str): The name of the sensor to retrieve.
        Returns:
            Optional[Sensor]: The Sensor object if found, otherwise None.
        """
        return self.sensors.get(sensor_name)

# Define sensor configurations for a typical farm plot
FARM_SENSORS_CONFIG: List[Dict[str, Any]] = [
    {'name': 'N', 'unit': 'ppm', 'min': 0, 'max': 140, 'initial_val': 70},
    {'name': 'P', 'unit': 'ppm', 'min': 0, 'max': 145, 'initial_val': 60},
    {'name': 'K', 'unit': 'ppm', 'min': 0, 'max': 205, 'initial_val': 90},
    {'name': 'temperature', 'unit': '°C', 'min': 0, 'max': 45, 'initial_val': 25},
    {'name': 'humidity', 'unit': '%', 'min': 0, 'max': 100, 'initial_val': 70},
    {'name': 'ph', 'unit': '', 'min': 0, 'max': 14, 'initial_val': 6.5},
    {'name': 'rainfall', 'unit': 'mm', 'min': 0, 'max': 300, 'initial_val': 100},
    {'name': 'soil_moisture', 'unit': '%', 'min': 0, 'max': 100, 'initial_val': 50},
    {'name': 'sunlight_exposure', 'unit': 'lux', 'min': 0, 'max': 140, 'initial_val': 70},
    {'name': 'co2_concentration', 'unit': 'ppm', 'min': 300, 'max': 1000, 'initial_val': 400},
    {'name': 'organic_matter', 'unit': '%', 'min': 0, 'max': 10, 'initial_val': 3.5},
]

# --- 2. Control Device Module ---
class ControlDevice:
    """
    Represents a control device that can perform actions in the farm ecosystem.
    """
    def __init__(self, name: str, action_func):
        self.name = name
        self.action_func = action_func
        self.status = "idle"
        self.lock = Lock()

    def perform_action(self, orchestrator_instance: 'Orchestrator', *args, **kwargs) -> str:
        """
        Executes the control device's action.
        Args:
            orchestrator_instance (Orchestrator): The orchestrator instance managing the plot.
            *args: Positional arguments for the action function.
            **kwargs: Keyword arguments for the action function.
        Returns:
            str: A message indicating the result of the action.
        """
        with self.lock:
            self.status = "active"
            farm_logger.info(f"[{self.name}] Performing action...")
            result = self.action_func(orchestrator_instance, *args, **kwargs)
            self.status = "idle"
            return result

    def get_status(self) -> str:
        """
        Retrieves the current status of the control device.
        Returns:
            str: The status of the device (e.g., "idle", "active").
        """
        with self.lock:
            return self.status

def water_crop_action(orchestrator_instance: 'Orchestrator', amount_ml: float) -> str:
    """
    Simulates watering the crop, increasing soil moisture.
    Args:
        orchestrator_instance (Orchestrator): The orchestrator managing the plot.
        amount_ml (float): The amount of water to apply in milliliters.
    Returns:
        str: A message confirming the action.
    """
    plot_id = orchestrator_instance.plot_id
    print(f"[ACTION] Plot {plot_id}: Watering crop with {amount_ml} ml of water.")
    soil_moisture_sensor = orchestrator_instance.sensor_group.get_sensor('soil_moisture')
    if soil_moisture_sensor:
        current_moisture = soil_moisture_sensor.get_value()
        new_moisture = min(soil_moisture_sensor.max_val, current_moisture + (amount_ml / 25.0))
        soil_moisture_sensor.set_value(new_moisture)
        print(f"[{plot_id}] Soil moisture increased to {new_moisture:.2f}%.")
    return f"Watered {plot_id} with {amount_ml} ml"

def notify_user_action(orchestrator_instance: 'Orchestrator', message: str) -> str:
    """
    Sends a notification message to the user.
    Args:
        orchestrator_instance (Orchestrator): The orchestrator managing the plot.
        message (str): The message to send.
    Returns:
        str: A message confirming the notification.
    """
    plot_id = orchestrator_instance.plot_id
    print(f"[ACTION] Plot {plot_id}: User Notification: {message}")
    return f"Notified user for {plot_id}: {message}"

def adjust_lighting_action(orchestrator_instance: 'Orchestrator', intensity_percent: float) -> str:
    """
    Simulates adjusting lighting, changing sunlight exposure.
    Args:
        orchestrator_instance (Orchestrator): The orchestrator managing the plot.
        intensity_percent (float): The desired lighting intensity (0-100%).
    Returns:
        str: A message confirming the action.
    """
    plot_id = orchestrator_instance.plot_id
    print(f"[ACTION] Plot {plot_id}: Adjusting lighting to {intensity_percent}%.")
    sunlight_sensor = orchestrator_instance.sensor_group.get_sensor('sunlight_exposure')
    if sunlight_sensor:
        new_sunlight = sunlight_sensor.min_val + (sunlight_sensor.max_val - sunlight_sensor.min_val) * (intensity_percent / 100.0)
        sunlight_sensor.set_value(new_sunlight)
        print(f"[{plot_id}] Sunlight exposure adjusted to {new_sunlight:.2f} lux.")
    return f"Adjusted lighting for {plot_id} to {intensity_percent}%"

def adjust_nutrient_action(orchestrator_instance: 'Orchestrator', nutrient_type: str, amount_mg: float) -> str:
    """
    Simulates adjusting nutrient levels (N, P, K) in the soil.
    Args:
        orchestrator_instance (Orchestrator): The orchestrator managing the plot.
        nutrient_type (str): The type of nutrient to adjust (e.g., 'NPK', 'N').
        amount_mg (float): The amount of nutrient to add in milligrams.
    Returns:
        str: A message confirming the action.
    """
    plot_id = orchestrator_instance.plot_id
    print(f"[ACTION] Plot {plot_id}: Adjusting {nutrient_type} by {amount_mg} mg.")
    if nutrient_type.upper() == 'NPK':
        n_sensor = orchestrator_instance.sensor_group.get_sensor('N')
        p_sensor = orchestrator_instance.sensor_group.get_sensor('P')
        k_sensor = orchestrator_instance.sensor_group.get_sensor('K')
        
        if n_sensor:
            n_sensor.set_value(min(n_sensor.max_val, n_sensor.get_value() + (amount_mg / 2.0)))
        if p_sensor:
            p_sensor.set_value(min(p_sensor.max_val, p_sensor.get_value() + (amount_mg / 2.0)))
        if k_sensor:
            k_sensor.set_value(min(k_sensor.max_val, k_sensor.get_value() + (amount_mg / 2.0)))
        print(f"[{plot_id}] NPK levels adjusted.")
    elif nutrient_type.upper() == 'N':
        n_sensor = orchestrator_instance.sensor_group.get_sensor('N')
        if n_sensor:
            n_sensor.set_value(min(n_sensor.max_val, n_sensor.get_value() + amount_mg))
        print(f"[{plot_id}] Nitrogen level adjusted.")
    return f"Adjusted {nutrient_type} for {plot_id} by {amount_mg} mg"

# --- 3. Crop Management Module ---
class Crop:
    """
    Represents a crop planted in a specific plot, tracking its type and parameters.
    """
    def __init__(self, plot_id: str, crop_type: str, current_growth_stage: str = "seedling"):
        self.plot_id = plot_id
        self.crop_type = crop_type
        self.current_growth_stage = current_growth_stage
        self.parameters: Dict[str, Union[int, float]] = {
            'target_yield': 100,
            'disease_risk': 0.1,
            'pest_risk': 0.05,
        }
        self.lock = Lock()

    def get_parameters(self) -> Dict[str, Union[int, float]]:
        """
        Retrieves a copy of the crop's current parameters.
        Returns:
            Dict[str, Union[int, float]]: A dictionary of crop parameters.
        """
        with self.lock:
            return self.parameters.copy()

    def update_parameter(self, param_name: str, value: Union[int, float]) -> bool:
        """
        Updates a specific parameter of the crop.
        Args:
            param_name (str): The name of the parameter to update.
            value (Union[int, float]): The new value for the parameter.
        Returns:
            bool: True if the parameter was updated, False otherwise.
        """
        with self.lock:
            if param_name in self.parameters:
                self.parameters[param_name] = value
                print(f"Crop {self.crop_type} (Plot {self.plot_id}): Parameter '{param_name}' updated to {value}")
                return True
            print(f"Crop {self.crop_type} (Plot {self.plot_id}): Parameter '{param_name}' not found.")
            return False

# --- 4. Orchestrator Module ---
class Orchestrator:
    """
    Manages a single farm plot, integrating sensors, control devices, and the recommendation system.
    Runs in a separate thread to continuously monitor conditions and trigger actions.
    """
    def __init__(self, plot_id: str, crop_type: str, sensor_group: SensorGroup, 
                 plot_control_devices: Dict[str, 'ControlDevice'], 
                 recommendation_system: RecommendationSystem, update_interval: int = 5):
        self.plot_id = plot_id
        self.crop = Crop(plot_id, crop_type)
        self.sensor_group = sensor_group
        self.control_devices = plot_control_devices
        self.recommendation_system = recommendation_system
        self.update_interval = update_interval
        self._running = False
        self._thread = None
        self._stop_event = Event()
        self.data_history: deque[tuple[float, Dict[str, float]]] = deque(maxlen=100)
        self.recommendation_history: deque[tuple[float, List[Dict[str, Any]]]] = deque(maxlen=20)

    def _run_orchestration_loop(self) -> None:
        """
        The main loop for the orchestrator. Reads sensor data, gets recommendations,
        and triggers control device actions.
        """
        while not self._stop_event.is_set():
            farm_logger.info(f"--- Orchestrator for Plot {self.plot_id} ({self.crop.crop_type}) ---")
            sensor_data = self.sensor_group.get_all_sensor_data()
            self.data_history.append((time.time(), sensor_data))
            farm_logger.info(f"Current Sensor Data: {sensor_data}")

            predicted_indicators = self.recommendation_system.predict_indicators(sensor_data, self.crop.crop_type)
            farm_logger.info(f"Predicted Indicators: {predicted_indicators}")

            # Pass current_sensor_data to the recommendation system for gradient analysis
            recommendations = self.recommendation_system.recommend_sensor_changes_from_indicators(
                predicted_indicators, self.crop.crop_type, sensor_data
            )
            self.recommendation_history.append((time.time(), recommendations))
            farm_logger.info("Recommendations:")
            for rec in recommendations:
                farm_logger.info(f"- {rec}")
                self._trigger_action_from_recommendation(rec)

            time.sleep(self.update_interval)

    def _trigger_action_from_recommendation(self, recommendation: Dict[str, Any]) -> None:
        """
        Interprets a structured recommendation and triggers the corresponding control device action.
        Args:
            recommendation (Dict[str, Any]): A dictionary containing the action type and parameters.
        """
        action_type = recommendation.get("action")
        message = recommendation.get("message")

        if action_type == "water_crop":
            amount_ml = recommendation.get("amount_ml")
            if amount_ml is not None and 'water_pump' in self.control_devices:
                self.control_devices['water_pump'].perform_action(self, amount_ml)
        elif action_type == "adjust_nutrient":
            nutrient_type = recommendation.get("nutrient_type")
            amount_mg = recommendation.get("amount_mg")
            if nutrient_type and amount_mg is not None and 'nutrient_dispenser' in self.control_devices:
                self.control_devices['nutrient_dispenser'].perform_action(self, nutrient_type, amount_mg)
        elif action_type == "adjust_lighting":
            intensity_percent = recommendation.get("intensity_percent")
            if intensity_percent is not None and 'lighting_system' in self.control_devices:
                self.control_devices['lighting_system'].perform_action(self, intensity_percent)
        elif action_type == "notify_user":
            if message and 'notify_user' in self.control_devices:
                self.control_devices['notify_user'].perform_action(self, message)
        else:
            print(f"[{self.plot_id}] Unknown recommendation action: {recommendation}")

    def start(self) -> None:
        """Starts the orchestrator's main loop in a separate thread."""
        if not self._running:
            self._running = True
            self._stop_event.clear()
            self._thread = Thread(target=self._run_orchestration_loop)
            self._thread.daemon = True
            self._thread.start()
            print(f"Orchestrator for Plot {self.plot_id} started.")

    def stop(self) -> None:
        """Stops the orchestrator's main loop and joins its thread."""
        if self._running:
            self._running = False
            self._stop_event.set()
            if self._thread:
                self._thread.join()
            print(f"Orchestrator for Plot {self.plot_id} stopped.")

# --- Main Simulation Setup ---
class SmartFarmEcosystem:
    """
    Manages multiple farm plots, each with its own sensors, control devices, and orchestrator.
    """
    def __init__(self):
        self.sensor_groups: Dict[str, SensorGroup] = {}
        self.orchestrators: Dict[str, Orchestrator] = {}
        self.lock = Lock()

    def add_farm_plot(self, plot_id: str, crop_type: str, 
                      plot_control_devices: Dict[str, ControlDevice], 
                      recommendation_system: RecommendationSystem,
                      sensors_config: List[Dict[str, Any]] = FARM_SENSORS_CONFIG, 
                      update_interval: int = 5) -> Optional[Orchestrator]:
        """
        Adds a new farm plot to the ecosystem, creating its sensor group and orchestrator.
        Args:
            plot_id (str): Unique identifier for the plot.
            crop_type (str): The type of crop planted in this plot.
            plot_control_devices (Dict[str, ControlDevice]): Dictionary of control devices for this plot.
            recommendation_system (RecommendationSystem): The instance of the recommendation system.
            sensors_config (List[Dict[str, Any]]): Configuration for the sensors in this plot.
            update_interval (int): How often the orchestrator loop runs in seconds.
        Returns:
            Optional[Orchestrator]: The created Orchestrator instance, or None if the plot already exists.
        """
        with self.lock:
            if plot_id in self.orchestrators:
                print(f"Plot {plot_id} already exists.")
                return None

            sensor_group = SensorGroup(f"Plot_{plot_id}_Sensors", sensors_config)
            orchestrator = Orchestrator(plot_id, crop_type, sensor_group, plot_control_devices, recommendation_system, update_interval)
            
            self.sensor_groups[plot_id] = sensor_group
            self.orchestrators[plot_id] = orchestrator
            print(f"Added Farm Plot {plot_id} with {crop_type}.")
            return orchestrator

    def start_all_orchestrators(self) -> None:
        """Starts all orchestrators managed by the ecosystem."""
        with self.lock:
            for orchestrator in self.orchestrators.values():
                orchestrator.start()

    def stop_all_orchestrators(self) -> None:
        """Stops all orchestrators managed by the ecosystem."""
        with self.lock:
            for orchestrator in self.orchestrators.values():
                orchestrator.stop()

    def get_plot_info(self, plot_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves comprehensive information about a specific farm plot.
        Args:
            plot_id (str): The ID of the plot to retrieve information for.
        Returns:
            Optional[Dict[str, Any]]: A dictionary containing plot details, or None if not found.
        """
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

    def get_sensor_for_plot(self, plot_id: str, sensor_name: str) -> Optional[Sensor]:
        """
        Retrieves a specific sensor from a given plot.
        Args:
            plot_id (str): The ID of the plot.
            sensor_name (str): The name of the sensor to retrieve.
        Returns:
            Optional[Sensor]: The Sensor object if found, otherwise None.
        """
        with self.lock:
            sensor_group = self.sensor_groups.get(plot_id)
            if sensor_group:
                return sensor_group.get_sensor(sensor_name)
            return None

    def update_crop_parameter(self, plot_id: str, param_name: str, value: Union[int, float]) -> bool:
        """
        Updates a crop parameter for a specific plot.
        Args:
            plot_id (str): The ID of the plot.
            param_name (str): The name of the crop parameter to update.
            value (Union[int, float]): The new value for the parameter.
        Returns:
            bool: True if the parameter was updated, False otherwise.
        """
        with self.lock:
            orchestrator = self.orchestrators.get(plot_id)
            if orchestrator:
                return orchestrator.crop.update_parameter(param_name, value)
            print(f"Plot {plot_id} not found.")
            return False

# Example Usage (to be run in a separate script or interactive session)
if __name__ == "__main__":
    print("This script defines the Smart Farm Ecosystem components.")
    print("To run a simulation, you would import these components and orchestrate them.")
    print("Example: ")
    print("  import smart_farm_ecosystem")
    print("  from smart_farm_recommendation import RecommendationSystem")
    print("  rec_system = RecommendationSystem()")
    print("  farm = smart_farm_ecosystem.SmartFarmEcosystem()")
    print("  # When adding a plot, pass the recommendation_system instance")
    print("  # farm.add_farm_plot('plot1', 'rice', plot_control_devices, rec_system)")
    print("  # farm.start_all_orchestrators()")
    print("  # To interact, e.g., change a sensor value:")
    print("  # sensor_n_plot1 = farm.get_sensor_for_plot('plot1', 'N')")
    print("  # if sensor_n_plot1: sensor_n_plot1.set_value(120)")
    print("  # To update a crop parameter:")
    print("  # farm.update_crop_parameter('plot1', 'target_yield', 150)")
    print("  # time.sleep(60) # Let it run for a while")
    print("  # farm.stop_all_orchestrators()")
