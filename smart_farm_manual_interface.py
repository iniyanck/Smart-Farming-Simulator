import pygame
import sys
from functools import partial

# Import necessary components from the smart farm ecosystem
from smart_farm_ecosystem import SmartFarmEcosystem, SensorGroup, ControlDevice, water_crop_action, adjust_nutrient_action, FARM_SENSORS_CONFIG
from smart_farm_recommendation import RecommendationSystem

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Smart Farm Manual Actions")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
BRIGHT_GREEN = (0, 255, 0)
RED = (200, 0, 0)
BRIGHT_RED = (255, 0, 0)
BLUE = (0, 0, 200)
BRIGHT_BLUE = (0, 0, 255)
LIGHT_GREY = (200, 200, 200)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
DARK_GREEN = (0, 100, 0)
DARK_RED = (100, 0, 0)

# Font
font = pygame.font.Font(None, 30) # Slightly smaller font for more info
large_font = pygame.font.Font(None, 36)

def text_objects(text, font, color=BLACK):
    text_surface = font.render(text, True, color)
    return text_surface, text_surface.get_rect()

def button(msg, x, y, w, h, ic, ac, action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    if x + w > mouse[0] > x and y + h > mouse[1] > y:
        pygame.draw.rect(screen, ac, (x, y, w, h))
        if click[0] == 1 and action is not None:
            action()
    else:
        pygame.draw.rect(screen, ic, (x, y, w, h))

    text_surf, text_rect = text_objects(msg, font)
    text_rect.center = ((x + (w / 2)), (y + (h / 2)))
    screen.blit(text_surf, text_rect)

def main_loop():
    # Initialize the recommendation system
    rec_system = RecommendationSystem()

    # Initialize the smart farm ecosystem
    farm_ecosystem = SmartFarmEcosystem()

    # Define multiple farm plots
    farm_plots_config = [
        {"plot_id": "plot1", "crop_type": "rice"},
        {"plot_id": "plot2", "crop_type": "maize"},
        {"plot_id": "plot3", "crop_type": "kidneybeans"},
        {"plot_id": "plot4", "crop_type": "pigeonpeas"},
    ]

    # Initialize control devices for all plots (these are generic and will be passed to each plot)
    generic_control_devices = {
        'water_pump': ControlDevice('water_pump', water_crop_action),
        'nutrient_dispenser': ControlDevice('nutrient_dispenser', adjust_nutrient_action),
    }

    # Add all farm plots to the ecosystem
    orchestrators = {}
    for plot_config in farm_plots_config:
        plot_id = plot_config["plot_id"]
        crop_type = plot_config["crop_type"]
        orchestrator = farm_ecosystem.add_farm_plot(plot_id, crop_type, generic_control_devices, rec_system, FARM_SENSORS_CONFIG, update_interval=10)
        if orchestrator:
            orchestrators[plot_id] = orchestrator
        else:
            print(f"Failed to add plot {plot_id} or it already exists.")

    if not orchestrators:
        print("No plots were successfully added. Exiting.")
        pygame.quit()
        sys.exit()

    farm_ecosystem.start_all_orchestrators()

    # Initial selected plot
    selected_plot_id = farm_plots_config[0]["plot_id"]

    def water_crop_manual(plot_id):
        print(f"Manual Action: Water Crop for {plot_id}")
        if plot_id in orchestrators and 'water_pump' in generic_control_devices:
            generic_control_devices['water_pump'].perform_action(orchestrators[plot_id], amount_ml=500)
        else:
            print(f"Water pump not available for {plot_id} or plot does not exist.")

    def add_fertilizer_manual(plot_id):
        print(f"Manual Action: Add Fertilizer for {plot_id}")
        if plot_id in orchestrators and 'nutrient_dispenser' in generic_control_devices:
            generic_control_devices['nutrient_dispenser'].perform_action(orchestrators[plot_id], nutrient_type='NPK', amount_mg=100)
        else:
            print(f"Nutrient dispenser not available for {plot_id} or plot does not exist.")

    def display_sensor_info(plot_id, current_sensors, predicted_indicators, ideal_ranges, crop_type, x_start, y_start):
        y_offset = y_start
        line_height = 30
        
        # Section Title
        title_surf, title_rect = text_objects(f"Plot: {plot_id.title()} ({crop_type.title()}) Status & Predictions", large_font, BLACK)
        title_rect.topleft = (x_start, y_offset)
        screen.blit(title_surf, title_rect)
        y_offset += line_height + 10

        # Current Sensor Readings
        sensor_title_surf, sensor_title_rect = text_objects("Current Sensor Readings:", font, BLACK)
        sensor_title_rect.topleft = (x_start, y_offset)
        screen.blit(sensor_title_surf, sensor_title_rect)
        y_offset += line_height

        for sensor_name, value in current_sensors.items():
            text = f"{sensor_name.replace('_', ' ').title()}: {value:.2f}"
            text_surf, text_rect = text_objects(text, font, BLACK)
            text_rect.topleft = (x_start + 20, y_offset)
            screen.blit(text_surf, text_rect)
            y_offset += line_height
        y_offset += 10

        # Predicted Indicators
        predicted_title_surf, predicted_title_rect = text_objects("Predicted Indicators:", font, BLACK)
        predicted_title_rect.topleft = (x_start, y_offset)
        screen.blit(predicted_title_surf, predicted_title_rect)
        y_offset += line_height

        for indicator, value in predicted_indicators.items():
            ideal_min, ideal_max = ideal_ranges.get(indicator, (None, None))
            color = BLACK
            status_text = ""
            range_text = ""

            if ideal_min is not None and ideal_max is not None:
                range_text = f" (Range: {ideal_min:.2f}-{ideal_max:.2f})"
                if value < ideal_min:
                    color = DARK_RED
                    status_text = " (LOW)"
                elif value > ideal_max:
                    color = DARK_RED
                    status_text = " (HIGH)"
                else:
                    color = DARK_GREEN
                    status_text = " (IDEAL)"
            
            text = f"{indicator}: {value:.2f}{status_text}{range_text}"
            text_surf, text_rect = text_objects(text, font, color)
            text_rect.topleft = (x_start + 20, y_offset)
            screen.blit(text_surf, text_rect)
            y_offset += line_height
        y_offset += 10

        # Removed the separate "Ideal Ranges" section as it's now integrated into Predicted Indicators

    def draw_plot_selection_buttons(plots_config, current_selected_plot_id, x_start, y_start, button_width, button_height, padding):
        plot_buttons = []
        y_offset = y_start
        for plot_config in plots_config:
            plot_id = plot_config["plot_id"]
            crop_type = plot_config["crop_type"]
            msg = f"{plot_id.title()} ({crop_type.title()})"
            
            rect = pygame.Rect(x_start, y_offset, button_width, button_height)
            
            if plot_id == current_selected_plot_id:
                pygame.draw.rect(screen, BRIGHT_BLUE, rect) # Highlight selected plot
            else:
                pygame.draw.rect(screen, BLUE, rect)
            
            text_surf, text_rect = text_objects(msg, font, WHITE)
            text_rect.center = rect.center
            screen.blit(text_surf, text_rect)
            
            plot_buttons.append((plot_id, rect))
            y_offset += button_height + padding
        return plot_buttons

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                # Check for plot selection clicks
                for plot_id, rect in plot_buttons:
                    if rect.collidepoint(mouse_pos):
                        selected_plot_id = plot_id
                        print(f"Selected plot: {selected_plot_id}")
                        break

        screen.fill(WHITE)

        # Title
        title_surf, title_rect = text_objects("Smart Farm Manual Actions", large_font)
        title_rect.center = (SCREEN_WIDTH / 2, 30)
        screen.blit(title_surf, title_rect)

        # Draw plot selection buttons
        plot_button_x = 50
        plot_button_y = 80
        plot_button_width = 200
        plot_button_height = 40
        plot_button_padding = 10
        plot_buttons = draw_plot_selection_buttons(farm_plots_config, selected_plot_id, plot_button_x, plot_button_y, plot_button_width, plot_button_height, plot_button_padding)

        # Manual action buttons for the selected plot
        if selected_plot_id:
            button("Water Crop", 50, plot_button_y + len(farm_plots_config) * (plot_button_height + plot_button_padding) + 20, 200, 50, GREEN, BRIGHT_GREEN, partial(water_crop_manual, selected_plot_id))
            button("Add Fertilizer", 50, plot_button_y + len(farm_plots_config) * (plot_button_height + plot_button_padding) + 90, 200, 50, BLUE, BRIGHT_BLUE, partial(add_fertilizer_manual, selected_plot_id))

            # Fetch current sensor data for the selected plot
            selected_orchestrator = orchestrators.get(selected_plot_id)
            if selected_orchestrator:
                current_sensor_data = selected_orchestrator.sensor_group.get_all_sensor_data()
                
                # Get crop type for the selected plot
                selected_crop_type = next((p["crop_type"] for p in farm_plots_config if p["plot_id"] == selected_plot_id), "unknown")

                # Predict indicators
                predicted_indicators = rec_system.predict_indicators(current_sensor_data, selected_crop_type)

                # Get ideal ranges for the current crop
                ideal_ranges = rec_system.crop_ideal_indicator_ranges.get(selected_crop_type, {})

                # Display sensor info, predictions, and ideal ranges for the selected plot
                display_sensor_info(selected_plot_id, current_sensor_data, predicted_indicators, ideal_ranges, selected_crop_type, 300, 80)
            else:
                text_surf, text_rect = text_objects(f"Error: Plot {selected_plot_id} not found.", font, RED)
                text_rect.topleft = (300, 80)
                screen.blit(text_surf, text_rect)
        else:
            text_surf, text_rect = text_objects("Select a plot to view its status.", font, BLACK)
            text_rect.topleft = (300, 80)
            screen.blit(text_surf, text_rect)


        pygame.display.update()

    farm_ecosystem.stop_all_orchestrators()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main_loop()
