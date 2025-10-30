import smart_farm_ecosystem
from smart_farm_recommendation import RecommendationSystem
import time
import sys
import argparse
from typing import Dict, Any

def main() -> None:
    """
    Main function to parse arguments and run a single Smart Farm Orchestrator process.
    """
    parser = argparse.ArgumentParser(description="Run a single Smart Farm Orchestrator.")
    parser.add_argument("--plot_id", required=True, help="ID of the farm plot.")
    parser.add_argument("--crop_type", required=True, help="Type of crop in the plot.")
    parser.add_argument("--update_interval", type=int, default=1, help="Update interval for the orchestrator in seconds.")
    args = parser.parse_args()

    plot_id: str = args.plot_id
    crop_type: str = args.crop_type
    update_interval: int = args.update_interval

    print(f"[{plot_id}] Initializing Orchestrator for Plot {plot_id} ({crop_type})...")

    print(f"[{plot_id}] Initializing Recommendation System...")
    try:
        recommendation_system = RecommendationSystem()
        recommendation_system.set_target_crop(crop_type)
        print(f"[{plot_id}] Recommendation System loaded successfully.")
    except Exception as e:
        print(f"[{plot_id}] Error initializing Recommendation System: {e}")
        print(f"[{plot_id}] Please ensure models are trained and available in the '{smart_farm_ecosystem.MODEL_DIR}' directory.")
        sys.exit(1)

    plot_control_devices: Dict[str, smart_farm_ecosystem.ControlDevice] = {
        'water_pump': smart_farm_ecosystem.ControlDevice('Water Pump', smart_farm_ecosystem.water_crop_action),
        'notify_user': smart_farm_ecosystem.ControlDevice('User Notifier', smart_farm_ecosystem.notify_user_action),
        'lighting_system': smart_farm_ecosystem.ControlDevice('Lighting System', smart_farm_ecosystem.adjust_lighting_action),
        'nutrient_dispenser': smart_farm_ecosystem.ControlDevice('Nutrient Dispenser', smart_farm_ecosystem.adjust_nutrient_action),
    }

    farm_instance = smart_farm_ecosystem.SmartFarmEcosystem()
    orchestrator = farm_instance.add_farm_plot(plot_id, crop_type, plot_control_devices, recommendation_system, update_interval=update_interval)

    if orchestrator:
        orchestrator.start()
        print(f"[{plot_id}] Orchestrator for Plot {plot_id} ({crop_type}) started in this window.")
        print(f"[{plot_id}] Type 'help' for commands specific to this orchestrator.")

        try:
            while True:
                command_input: List[str] = input(f"\n[{plot_id}] Enter command (e.g., 'set N 100', 'crop_param target_yield 150', 'exit'): ").strip().lower().split(maxsplit=2)
                
                if not command_input:
                    continue

                action: str = command_input[0]

                if action == 'exit':
                    print(f"[{plot_id}] Stopping orchestrator...")
                    break
                elif action == 'help':
                    print(f"\n[{plot_id}] Available Commands for Orchestrator {plot_id}:")
                    print(f"  crop_param [param_name] [value] - Update a crop parameter (e.g., 'crop_param target_yield 150')")
                    print(f"  status                  - Show current sensor data, crop parameters, and recent history")
                    print(f"  exit                    - Stop this orchestrator")
                    print(f"  help                    - Show this help message")
                    print(f"\n[{plot_id}] Note: Sensor adjustments are now solely driven by the Recommendation System's analysis.")
                elif action == 'crop_param':
                    if len(command_input) < 3:
                        print(f"[{plot_id}] Usage: crop_param [param_name] [value]")
                        continue
                    param_name: str = command_input[1]
                    try:
                        value: float = float(command_input[2])
                        if farm_instance.update_crop_parameter(plot_id, param_name, value):
                            print(f"[{plot_id}] Updated crop parameter '{param_name}' to {value}.")
                        else:
                            print(f"[{plot_id}] Failed to update crop parameter '{param_name}'.")
                    except ValueError:
                        print(f"[{plot_id}] Invalid value. Please enter a number.")
                elif action == 'status':
                    info: Optional[Dict[str, Any]] = farm_instance.get_plot_info(plot_id)
                    if info:
                        print(f"\n--- [{plot_id}] Status for Plot {plot_id} ({info['crop_type']}) ---")
                        print("  Current Sensor Data:")
                        for sensor_name, value in info['sensor_data'].items():
                            print(f"    {sensor_name}: {value}")
                        print("  Crop Parameters:")
                        for param_name, value in info['crop_parameters'].items():
                            print(f"    {param_name}: {value}")
                        print("  Control Device Statuses:")
                        for dev_name, dev_status in info['control_device_statuses'].items():
                            print(f"    {dev_name}: {dev_status}")
                        
                        print("\n  Recent Sensor Data History (last 5):")
                        if orchestrator.data_history:
                            for i, (timestamp, data) in enumerate(list(orchestrator.data_history)[-5:]):
                                print(f"    {i+1}. {time.strftime('%H:%M:%S', time.localtime(timestamp))}: {data}")
                        else:
                            print("    No sensor data history available.")

                        print("\n  Recent Recommendation History (last 3):")
                        if orchestrator.recommendation_history:
                            for i, (timestamp, recs) in enumerate(list(orchestrator.recommendation_history)[-3:]):
                                print(f"    {i+1}. {time.strftime('%H:%M:%S', time.localtime(timestamp))}:")
                                for rec in recs:
                                    print(f"      - {rec}")
                        else:
                            print("    No recommendation history available.")
                    else:
                        print(f"[{plot_id}] Plot {plot_id} not found in this orchestrator's context.")
                else:
                    print(f"[{plot_id}] Unknown command: '{action}'. Type 'help' for available commands.")
                
                time.sleep(0.1)

        except KeyboardInterrupt:
            print(f"\n[{plot_id}] Ctrl+C detected. Stopping Orchestrator for Plot {plot_id}...")
        finally:
            orchestrator.stop()
            print(f"[{plot_id}] Orchestrator for Plot {plot_id} stopped.")
    else:
        print(f"[{plot_id}] Failed to initialize orchestrator for Plot {plot_id}.")
        sys.exit(1)

if __name__ == "__main__":
    main()
