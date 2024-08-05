import csv
import random

# Generate 1000 rows of mock data
data = []
for _ in range(1000):
    air_temperature = round(random.uniform(25.0, 32.0), 1)
    nitrogen_level = round(random.uniform(0.3, 0.9), 1)
    soil_moisture_level = round(random.uniform(0.4, 0.8), 1)
    light_sensor_reading = random.randint(115, 130)
    label = random.choice([1, 0]) # 1 - Water, 0 - Do not Water
    # row = [air_temperature, nitrogen_level, soil_moisture_level, light_sensor_reading, label]
    row = [air_temperature, soil_moisture_level, light_sensor_reading, label]
    data.append(row)

# Save data to CSV file
filename = "plant_data.csv"
with open(filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # writer.writerow(["airTemperature", "nitrogenLevel", "soilMoistureLevel", "lightSensorReading", "Label"])
    writer.writerow(["airTemperature", "soilMoistureLevel", "lightSensorReading", "label"])
    writer.writerows(data)

print(f"Data saved to {filename} successfully.")
