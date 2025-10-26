import smart_farm_ecosystem
import smart_farm_recommendation as rec_sys
import time
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run a single Smart Farm Orchestrator.")
    parser.add_argument("--plot_id", required=True, help="ID of the farm plot.")
    parser.add_argument("--crop_type", required=True, help="Type of crop in the plot.")
    parser.add_argument("--update_interval", type=int, default=5, help="Update interval for the orchestrator in seconds.")
    args = parser.parse_args()

    plot_id = args.plot_id
    crop_type = args.crop_type
    update_interval = args.update_interval

    print(f"[{plot_id}] Initializing Orchestrator for Plot {plot_id} ({crop_type})...")

    # Load recommendation system components
    try:
        smart_farm_ecosystem.load_recommendation_system_components(rec_sys)
    except Exception as e:
        print(f"[{plot_id}] Error loading recommendation system components: {e}")
        print(f"[{plot_id}] Please ensure smart_farm_recommendation.py has been run at least once to train models and define global variables.")
        sys.exit(1)

    # Create a dummy SmartFarmEcosystem to hold the single orchestrator and its sensors/devices
    # This is a simplified setup for a single process.
    # In a truly distributed system, sensors and control devices might also be separate processes.
    # Create plot-specific control devices
    plot_control_devices = {
        'water_pump': smart_farm_ecosystem.ControlDevice('Water Pump', smart_farm_ecosystem.water_crop_action),
        'notify_user': smart_farm_ecosystem.ControlDevice('User Notifier', smart_farm_ecosystem.notify_user_action),
        'lighting_system': smart_farm_ecosystem.ControlDevice('Lighting System', smart_farm_ecosystem.adjust_lighting_action),
        'nutrient_dispenser': smart_farm_ecosystem.ControlDevice('Nutrient Dispenser', smart_farm_ecosystem.adjust_nutrient_action),
    }

    farm_instance = smart_farm_ecosystem.SmartFarmEcosystem()
    orchestrator = farm_instance.add_farm_plot(plot_id, crop_type, plot_control_devices, update_interval=update_interval)

    if orchestrator:
        orchestrator.start()
        print(f"[{plot_id}] Orchestrator for Plot {plot_id} ({crop_type}) started in this window.")
        print(f"[{plot_id}] Type 'help' for commands specific to this orchestrator.")

        try:
            while True:
                command_input = input(f"\n[{plot_id}] Enter command (e.g., 'set N 100', 'crop_param target_yield 150', 'exit'): ").strip().lower().split(maxsplit=2)
                
                if not command_input:
                    continue

                action = command_input[0]

                if action == 'exit':
                    print(f"[{plot_id}] Stopping orchestrator...")
                    break
                elif action == 'help':
                    print(f"\n[{plot_id}] Available Commands for Orchestrator {plot_id}:")
                    print(f"  set [sensor_name] [value] - Manually set a sensor value (e.g., 'set N 120')")
                    print(f"  crop_param [param_name] [value] - Update a crop parameter (e.g., 'crop_param target_yield 150')")
                    print(f"  status                  - Show current sensor data and crop parameters for this plot")
                    print(f"  exit                    - Stop this orchestrator")
                    print(f"  help                    - Show this help message")
                elif action == 'set':
                    if len(command_input) < 3:
                        print(f"[{plot_id}] Usage: set [sensor_name] [value]")
                        continue
                    sensor_name = command_input[1]
                    try:
                        value = float(command_input[2])
                        sensor = farm_instance.get_sensor_for_plot(plot_id, sensor_name)
                        if sensor:
                            sensor.set_value(value)
                        else:
                            print(f"[{plot_id}] Sensor '{sensor_name}' not found for plot '{plot_id}'.")
                    except ValueError:
                        print(f"[{plot_id}] Invalid value. Please enter a number.")
                elif action == 'crop_param':
                    if len(command_input) < 3:
                        print(f"[{plot_id}] Usage: crop_param [param_name] [value]")
                        continue
                    param_name = command_input[1]
                    try:
                        value = float(command_input[2])
                        if farm_instance.update_crop_parameter(plot_id, param_name, value):
                            print(f"[{plot_id}] Updated crop parameter '{param_name}' to {value}.")
                        else:
                            print(f"[{plot_id}] Failed to update crop parameter '{param_name}'.")
                    except ValueError:
                        print(f"[{plot_id}] Invalid value. Please enter a number.")
                elif action == 'status':
                    info = farm_instance.get_plot_info(plot_id)
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
                        print(f"  Sensor Data History Count: {len(orchestrator.data_history)}")
                        print(f"  Recommendation History Count: {len(orchestrator.recommendation_history)}")
                    else:
                        print(f"[{plot_id}] Plot {plot_id} not found in this orchestrator's context.")
                else:
                    print(f"[{plot_id}] Unknown command: '{action}'. Type 'help' for available commands.")
                
                time.sleep(0.1) # Small delay to prevent busy-waiting in the loop

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
