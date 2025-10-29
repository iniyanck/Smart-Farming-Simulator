import smart_farm_ecosystem
from smart_farm_recommendation import RecommendationSystem
import time
import sys
import os
import subprocess
import platform
from multiprocessing import Lock
from typing import Dict, Any, Optional, List

def clear_screen() -> None:
    """Clears the terminal screen based on the operating system."""
    os.system('cls' if os.name == 'nt' else 'clear')

class MultiProcessSmartFarmEcosystem:
    """
    Manages the launching and stopping of multiple Smart Farm Orchestrator processes.
    Each orchestrator runs in a separate terminal window.
    """
    def __init__(self):
        self.orchestrator_processes: Dict[str, subprocess.Popen] = {}
        self.plot_configs: Dict[str, Dict[str, Any]] = {}
        self.lock = Lock()

    def add_farm_plot(self, plot_id: str, crop_type: str, update_interval: int = 5) -> bool:
        """
        Configures a new farm plot to be launched.
        Args:
            plot_id (str): Unique identifier for the plot.
            crop_type (str): The type of crop in the plot.
            update_interval (int): The update interval for the orchestrator in seconds.
        Returns:
            bool: True if the plot was added, False if it already exists.
        """
        with self.lock:
            if plot_id in self.orchestrator_processes:
                print(f"Plot {plot_id} already exists.")
                return False
            self.plot_configs[plot_id] = {'crop_type': crop_type, 'update_interval': update_interval}
            print(f"Configured Farm Plot {plot_id} with {crop_type}.")
            return True

    def launch_orchestrator_process(self, plot_id: str) -> Optional[subprocess.Popen]:
        """
        Launches a single orchestrator process for a given plot in a new terminal window.
        Args:
            plot_id (str): The ID of the plot to launch.
        Returns:
            Optional[subprocess.Popen]: The Popen object for the launched process, or None if an error occurs.
        """
        if plot_id not in self.plot_configs:
            print(f"Error: Plot {plot_id} not configured.")
            return None

        crop_type = self.plot_configs[plot_id]['crop_type']
        update_interval = self.plot_configs[plot_id]['update_interval']

        command: List[str] = [
            sys.executable,
            "orchestrator_runner.py",
            "--plot_id", plot_id,
            "--crop_type", crop_type,
            "--update_interval", str(update_interval)
        ]

        process: Optional[subprocess.Popen] = None
        if platform.system() == "Windows":
            full_command = ["start", "cmd", "/k"] + command
            process = subprocess.Popen(full_command, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
        elif platform.system() == "Darwin": # macOS
            print(f"On macOS, please open a new terminal and run: {' '.join(command)}")
            process = subprocess.Popen(command, preexec_fn=os.setpgrp)
        else: # Linux and other Unix-like systems
            print(f"On Linux, please open a new terminal and run: {' '.join(command)}")
            process = subprocess.Popen(command, preexec_fn=os.setpgrp)
        
        with self.lock:
            self.orchestrator_processes[plot_id] = process
        print(f"Launched orchestrator for Plot {plot_id} in a new terminal (or instructed to do so).")
        return process

    def start_all_orchestrators(self) -> None:
        """Launches orchestrator processes for all configured plots."""
        with self.lock:
            for plot_id in self.plot_configs:
                self.launch_orchestrator_process(plot_id)

    def stop_all_orchestrators(self) -> None:
        """Terminates all running orchestrator processes."""
        with self.lock:
            print("\nStopping all orchestrator processes...")
            for plot_id, process in self.orchestrator_processes.items():
                if process.poll() is None:
                    print(f"Terminating orchestrator for Plot {plot_id} (PID: {process.pid})...")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print(f"Orchestrator for Plot {plot_id} did not terminate gracefully. Killing...")
                        process.kill()
                print(f"Orchestrator for Plot {plot_id} stopped.")
            self.orchestrator_processes.clear()
            print("All orchestrator processes stopped.")

    def get_plot_status(self, plot_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the process status for a specific plot's orchestrator.
        Args:
            plot_id (str): The ID of the plot.
        Returns:
            Optional[Dict[str, Any]]: A dictionary with status information, or None if not found.
        """
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

def main() -> None:
    """
    Main function for the Smart Farm Ecosystem Launcher.
    Initializes the recommendation system, configures plots, launches orchestrators,
    and provides an interactive command-line interface.
    """
    clear_screen()
    print("Initializing Smart Farm Ecosystem Launcher...")

    print("Initializing Recommendation System to ensure models are trained/loaded...")
    try:
        _ = RecommendationSystem() 
        print("Recommendation System models are ready.")
    except Exception as e:
        print(f"Error during Recommendation System initialization: {e}")
        sys.exit(1)

    farm_launcher = MultiProcessSmartFarmEcosystem()

    farm_launcher.add_farm_plot('plot1', 'rice', update_interval=3)
    farm_launcher.add_farm_plot('plot2', 'maize', update_interval=4)
    farm_launcher.add_farm_plot('plot3', 'coffee', update_interval=5)

    farm_launcher.start_all_orchestrators()

    print("\nSmart Farm Simulation Launched!")
    print("Each orchestrator is running in its own terminal window.")
    print("Interact with each orchestrator directly in its respective terminal for sensor/crop parameter changes.")
    print("Type 'help' for commands in this launcher window.")

    try:
        while True:
            command: List[str] = input("\nEnter command (e.g., 'status plot1', 'list_plots', 'exit'): ").strip().lower().split()
            
            if not command:
                continue

            action: str = command[0]

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
                plot_id: str = command[1]
                info: Optional[Dict[str, Any]] = farm_launcher.get_plot_status(plot_id)
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
                    for plot_id_key in farm_launcher.plot_configs:
                        status_info: Optional[Dict[str, Any]] = farm_launcher.get_plot_status(plot_id_key)
                        if status_info:
                            print(f"- {plot_id_key} (Crop: {status_info['crop_type']}, Process Status: {status_info['orchestrator_status']}, PID: {status_info['pid'] if status_info['pid'] else 'N/A'})")
                        else:
                            print(f"- {plot_id_key} (Configured, but process not found or stopped)")
            else:
                print(f"Unknown command: '{action}'. Type 'help' for available commands.")
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nCtrl+C detected in launcher. Stopping all orchestrators...")
    finally:
        farm_launcher.stop_all_orchestrators()
        print("Smart Farm Simulation Launcher Exited.")

if __name__ == "__main__":
    main()
