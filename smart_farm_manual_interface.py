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

    # Define a plot ID and crop type
    plot_id = "plot1"
    crop_type = "rice"

    # Define control devices for the plot
    plot_control_devices = {
        'water_pump': ControlDevice('water_pump', water_crop_action),
        'nutrient_dispenser': ControlDevice('nutrient_dispenser', adjust_nutrient_action),
        # Add other control devices as needed, e.g., for lighting, harvesting
    }

    # Add the farm plot to the ecosystem
    orchestrator = farm_ecosystem.add_farm_plot(plot_id, crop_type, plot_control_devices, rec_system, FARM_SENSORS_CONFIG, update_interval=10)
    if orchestrator:
        farm_ecosystem.start_all_orchestrators()
    else:
        print(f"Failed to add plot {plot_id} or it already exists.")
        pygame.quit()
        sys.exit()

    def water_crop_manual():
        print("Manual Action: Water Crop")
        if 'water_pump' in plot_control_devices:
            plot_control_devices['water_pump'].perform_action(orchestrator, amount_ml=500)
        else:
            print("Water pump not available.")

    def add_fertilizer_manual():
        print("Manual Action: Add Fertilizer")
        if 'nutrient_dispenser' in plot_control_devices:
            plot_control_devices['nutrient_dispenser'].perform_action(orchestrator, nutrient_type='NPK', amount_mg=100)
        else:
            print("Nutrient dispenser not available.")

    def display_sensor_info(current_sensors, predicted_indicators, ideal_ranges, crop_type, x_start, y_start):
        y_offset = y_start
        line_height = 30
        
        # Section Title
        title_surf, title_rect = text_objects("Farm Status & Predictions", large_font, BLACK)
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

            if ideal_min is not None and ideal_max is not None:
                if value < ideal_min:
                    color = DARK_RED
                    status_text = " (LOW)"
                elif value > ideal_max:
                    color = DARK_RED
                    status_text = " (HIGH)"
                else:
                    color = DARK_GREEN
                    status_text = " (IDEAL)"
            
            text = f"{indicator}: {value:.2f}{status_text}"
            text_surf, text_rect = text_objects(text, font, color)
            text_rect.topleft = (x_start + 20, y_offset)
            screen.blit(text_surf, text_rect)
            y_offset += line_height
        y_offset += 10

        # Ideal Ranges
        ideal_title_surf, ideal_title_rect = text_objects(f"Ideal Ranges for {crop_type.title()}:", font, BLACK)
        ideal_title_rect.topleft = (x_start, y_offset)
        screen.blit(ideal_title_surf, ideal_title_rect)
        y_offset += line_height

        for indicator, (min_val, max_val) in ideal_ranges.items():
            text = f"{indicator}: {min_val:.2f} - {max_val:.2f}"
            text_surf, text_rect = text_objects(text, font, BLACK)
            text_rect.topleft = (x_start + 20, y_offset)
            screen.blit(text_surf, text_rect)
            y_offset += line_height

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)

        # Title
        title_surf, title_rect = text_objects("Smart Farm Manual Actions", large_font)
        title_rect.center = (SCREEN_WIDTH / 2, 30)
        screen.blit(title_surf, title_rect)

        # Buttons for manual actions
        button("Water Crop", 50, 100, 200, 50, GREEN, BRIGHT_GREEN, water_crop_manual)
        button("Add Fertilizer", 50, 170, 200, 50, BLUE, BRIGHT_BLUE, add_fertilizer_manual)

        # Fetch current sensor data
        current_sensor_data = orchestrator.sensor_group.get_all_sensor_data()
        
        # Predict indicators
        predicted_indicators = rec_system.predict_indicators(current_sensor_data)

        # Get ideal ranges for the current crop
        ideal_ranges = rec_system.crop_ideal_indicator_ranges.get(crop_type, {})

        # Display sensor info, predictions, and ideal ranges
        display_sensor_info(current_sensor_data, predicted_indicators, ideal_ranges, crop_type, 300, 80)

        pygame.display.update()

    farm_ecosystem.stop_all_orchestrators()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main_loop()
