import smart_farm_ecosystem
import smart_farm_recommendation as rec_sys
import time
import sys
import os
import subprocess
import platform

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

class MultiProcessSmartFarmEcosystem:
    def __init__(self):
        self.orchestrator_processes = {} # plot_id -> Popen object
        self.plot_configs = {} # plot_id -> {'crop_type': ..., 'update_interval': ...}
        self.lock = Lock() # For managing access to shared state if needed, though less critical with separate processes

    def add_farm_plot(self, plot_id, crop_type, update_interval=5):
        with self.lock:
            if plot_id in self.orchestrator_processes:
                print(f"Plot {plot_id} already exists.")
                return False
            self.plot_configs[plot_id] = {'crop_type': crop_type, 'update_interval': update_interval}
            print(f"Configured Farm Plot {plot_id} with {crop_type}.")
            return True

    def launch_orchestrator_process(self, plot_id):
        if plot_id not in self.plot_configs:
            print(f"Error: Plot {plot_id} not configured.")
            return

        crop_type = self.plot_configs[plot_id]['crop_type']
        update_interval = self.plot_configs[plot_id]['update_interval']

        command = [
            sys.executable, # Path to the current Python interpreter
            "orchestrator_runner.py",
            "--plot_id", plot_id,
            "--crop_type", crop_type,
            "--update_interval", str(update_interval)
        ]

        # Determine how to open a new terminal window based on OS
        if platform.system() == "Windows":
            # Use 'start cmd /k' to open a new cmd window and keep it open after command
            full_command = ["start", "cmd", "/k"] + command
            process = subprocess.Popen(full_command, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
        elif platform.system() == "Darwin": # macOS
            # Use 'open -a Terminal' or 'open -a iTerm'
            # This might require user interaction or specific terminal settings
            # For simplicity, we'll just run it in the background for now,
            # or the user can manually open new terminals and run the command.
            # A more robust solution would involve AppleScript.
            print(f"On macOS, please open a new terminal and run: {' '.join(command)}")
            process = subprocess.Popen(command, preexec_fn=os.setpgrp) # Detach process
        else: # Linux and other Unix-like systems
            # Use 'xterm -e' or 'gnome-terminal -e' or 'konsole -e'
            # This depends on the desktop environment. xterm is usually a safe bet.
            # For simplicity, we'll just run it in the background for now.
            print(f"On Linux, please open a new terminal and run: {' '.join(command)}")
            process = subprocess.Popen(command, preexec_fn=os.setpgrp) # Detach process
        
        with self.lock:
            self.orchestrator_processes[plot_id] = process
        print(f"Launched orchestrator for Plot {plot_id} in a new terminal (or instructed to do so).")
        return process

    def start_all_orchestrators(self):
        with self.lock:
            for plot_id in self.plot_configs:
                self.launch_orchestrator_process(plot_id)

    def stop_all_orchestrators(self):
        with self.lock:
            print("\nStopping all orchestrator processes...")
            for plot_id, process in self.orchestrator_processes.items():
                if process.poll() is None: # If process is still running
                    print(f"Terminating orchestrator for Plot {plot_id} (PID: {process.pid})...")
                    if platform.system() == "Windows":
                        # On Windows, terminate() might not close the console window.
                        # We might need taskkill or a more aggressive approach if terminate() isn't enough.
                        process.terminate()
                    else:
                        # On Unix-like systems, send SIGTERM
                        process.terminate()
                    process.wait(timeout=5) # Give it some time to terminate
                    if process.poll() is None:
                        print(f"Orchestrator for Plot {plot_id} did not terminate gracefully. Killing...")
                        process.kill()
                print(f"Orchestrator for Plot {plot_id} stopped.")
            self.orchestrator_processes.clear()
            print("All orchestrator processes stopped.")

    def get_plot_status(self, plot_id):
        with self.lock:
            if plot_id in self.orchestrator_processes:
                process = self.orchestrator_processes[plot_id]
                status = "Running" if process.poll() is None else "Stopped"
                config = self.plot_configs[plot_id]
                return {
                    'plot_id': plot_id,
                    'crop_type': config['crop_type'],
                    'orchestrator_status': status,
                    'pid': process.pid if process.poll() is None else None
                }
            return None

def main():
    clear_screen()
    print("Initializing Smart Farm Ecosystem Launcher...")

    # Load recommendation system components (only needed for training, not for launching)
    # The orchestrator_runner.py script will load them individually.
    # However, it's good practice to ensure the models are trained once.
    print("Ensuring recommendation models are trained (running smart_farm_recommendation.py)...")
    try:
        # Run smart_farm_recommendation.py in a non-interactive way
        subprocess.run([sys.executable, "smart_farm_recommendation.py"], check=True, capture_output=True)
        print("Recommendation models trained successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error training recommendation models: {e}")
        print(f"Stdout: {e.stdout.decode()}")
        print(f"Stderr: {e.stderr.decode()}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during model training: {e}")
        sys.exit(1)

    farm_launcher = MultiProcessSmartFarmEcosystem()

    # Add multiple farm plots
    farm_launcher.add_farm_plot('plot1', 'rice', update_interval=3)
    farm_launcher.add_farm_plot('plot2', 'maize', update_interval=4)
    farm_launcher.add_farm_plot('plot3', 'coffee', update_interval=5)

    # Start all orchestrators in separate processes/windows
    farm_launcher.start_all_orchestrators()

    print("\nSmart Farm Simulation Launched!")
    print("Each orchestrator is running in its own terminal window.")
    print("Interact with each orchestrator directly in its respective terminal for sensor/crop parameter changes.")
    print("Type 'help' for commands in this launcher window.")

    try:
        while True:
            command = input("\nEnter command (e.g., 'status plot1', 'list_plots', 'exit'): ").strip().lower().split()
            
            if not command:
                continue

            action = command[0]

            if action == 'exit':
                print("Stopping simulation launcher...")
                break
            elif action == 'help':
                print("\nAvailable Commands for Launcher:")
                print("  status [plot_id]        - Show process status of a specific plot's orchestrator (e.g., 'status plot1')")
                print("  list_plots              - List all active farm plots and their process status")
                print("  exit                    - Stop all orchestrators and exit the launcher")
                print("  help                    - Show this help message")
                print("\nTo change sensor values or crop parameters, go to the individual orchestrator's terminal window.")
                print("Each orchestrator terminal will have its own prompt for interaction (if implemented in orchestrator_runner.py).")
            elif action == 'status':
                if len(command) < 2:
                    print("Usage: status [plot_id]")
                    continue
                plot_id = command[1]
                info = farm_launcher.get_plot_status(plot_id)
                if info:
                    print(f"\n--- Launcher Status for Plot {plot_id} ({info['crop_type']}) ---")
                    print(f"  Orchestrator Process Status: {info['orchestrator_status']}")
                    print(f"  PID: {info['pid']}")
                    print(f"  For detailed sensor data and recommendations, check the dedicated terminal for Plot {plot_id}.")
                else:
                    print(f"Plot {plot_id} not found or not launched by this launcher.")
            elif action == 'list_plots':
                if not farm_launcher.orchestrator_processes:
                    print("No farm plots currently active.")
                else:
                    print("\nActive Farm Plots (managed by this launcher):")
                    for plot_id in farm_launcher.plot_configs:
                        status_info = farm_launcher.get_plot_status(plot_id)
                        if status_info:
                            print(f"- {plot_id} (Crop: {status_info['crop_type']}, Process Status: {status_info['orchestrator_status']}, PID: {status_info['pid'] if status_info['pid'] else 'N/A'})")
                        else:
                            print(f"- {plot_id} (Configured, but process not found or stopped)")
            else:
                print(f"Unknown command: '{action}'. Type 'help' for available commands.")
            
            time.sleep(0.1) # Small delay to prevent busy-waiting in the loop

    except KeyboardInterrupt:
        print("\nCtrl+C detected in launcher. Stopping all orchestrators...")
    finally:
        farm_launcher.stop_all_orchestrators()
        print("Smart Farm Simulation Launcher Exited.")

if __name__ == "__main__":
    main()
