from flask import Flask, request, jsonify
import joblib
import base64
import io
import os
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import base64

from plantGroupDetectionModel import detect_object, split_image
from teachableMachine import teachableMachinePython


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("./teachablemachine/keras_model.h5", compile=False)

# Load the labels
class_names = open("./teachablemachine/labels.txt", "r").readlines()


# Helper functions
def base64_to_png(base64_string, output_file):
    try:
        # Decode the base64 string
        image_data = base64.b64decode(base64_string)

        # Create a file-like object from the decoded image data
        image_buffer = io.BytesIO(image_data)

        # Open the image using PIL
        image = Image.open(image_buffer)

        # Save the image as PNG
        image.save(output_file, "PNG")

        print(f"Successfully converted base64 to PNG. Saved as {output_file}")

    except Exception as e:
        print("Error occurred while converting base64 to PNG:", str(e))


app = Flask(__name__)

# Load the saved model
linear_regression_model = joblib.load("./models/linear_regression_model.pkl")
decision_tree_model = joblib.load("./models/decision_tree_model.pkl")


@app.route("/waterPlant_LR", methods=["GET"])
def handle_get_request_waterPlant_LR():
    air_temperature = float(request.args.get("airTemperature"))
    soil_moisture_level = float(request.args.get("soilMoistureLevel"))
    light_sensor_reading = float(request.args.get("lightSensorReading"))

    # Predict the label using the model
    input_data = [[air_temperature, soil_moisture_level, light_sensor_reading]]
    prediction = linear_regression_model.predict(input_data)[0]

    # Return the appropriate response based on the prediction
    if prediction == 1:
        result = "Water Plant"
    else:
        result = "Do not Water Plant"

    return result


@app.route("/waterPlant_DT", methods=["GET"])
def handle_get_request_waterPlantDT():
    air_temperature = float(request.args.get("airTemperature"))
    soil_moisture_level = float(request.args.get("soilMoistureLevel"))
    light_sensor_reading = float(request.args.get("lightSensorReading"))

    # Predict the label using the model
    input_data = [[air_temperature, soil_moisture_level, light_sensor_reading]]
    prediction = decision_tree_model.predict(input_data)[0]

    # Return the appropriate response based on the prediction
    if prediction == 1:
        result = "Water Plant"
    else:
        result = "Do not Water Plant"

    return result


@app.route("/plantDetection", methods=["POST"])
def convert_base64_to_png():
    data = request.get_json()
    try:
        # Get the base64 string from the request body
        image_base64 = data["imageBase64"]
        # Decode the base64 string
        image_data = base64.b64decode(image_base64)

        # Create a file-like object from the decoded image data
        image_buffer = io.BytesIO(image_data)

        # Open the image using PIL
        image = Image.open(image_buffer)

        # Save the image as PNG
        # output_file = "./plantPhotos/photo1.png"
        # image.save(output_file, "PNG")
        output_file = "./plantPhotos/photo1.jpg"
        image.save(output_file, "JPEG")

        rawImagePath = output_file
        # split image
        split_image(rawImagePath)
        # check left
        leftGroupPath = "./plantGroupDetection/splittedPhotos/left.png"
        rightGroupPath = "./plantGroupDetection/splittedPhotos/right.png"

        leftGroupWhiteDots = detect_object("left_group", leftGroupPath)
        leftGroupResult = teachableMachinePython(model, class_names, leftGroupPath)
        # check right
        rightGroupWhiteDots = detect_object(
            "right_group", "./plantGroupDetection/splittedPhotos/right.png"
        )
        rightGroupResult = teachableMachinePython(model, class_names, rightGroupPath)

        if rightGroupWhiteDots > leftGroupWhiteDots:
            rightGroup = "Group 1"
            leftGroup = "Group 2"
        else:
            rightGroup = "Group 2"
            leftGroup = "Group 1"

        # rightModelResult
        print("done")

        return leftGroupResult + ";" + rightGroupResult
        # return {
        #     "leftGroup": [leftGroup, leftGroupResult],
        #     "rightGroup": [rightGroup, rightGroupResult],
        # }

        # return f"Image successfully converted and saved as {output_file}"

    except Exception as e:
        print("error")
        print(e)
        return f"Error occurred while converting base64 to PNG: {str(e)}"


@app.route("/plantDetection_Test", methods=["POST"])
def testLogic():
    data = request.get_json()
    # rawImagePath = "./plantGroupDetection/plantGroupTesting/diamond1.jpg"
    # rawImagePath = "./plantGroupDetection/plantGroupTesting/diamond2.jpg"
    # rawImagePath = "./plantGroupDetection/plantGroupTesting/heart1.jpg"
    rawImagePath = "./plantGroupDetection/plantGroupTesting/heart2.jpg"

    # split image
    split_image(rawImagePath)
    # check left
    leftGroupPath = "./plantGroupDetection/splittedPhotos/left.png"
    rightGroupPath = "./plantGroupDetection/splittedPhotos/right.png"

    leftGroupWhiteDots = detect_object("left_group", leftGroupPath)
    leftGroupResult = teachableMachinePython(model, class_names, leftGroupPath)
    # check right
    rightGroupWhiteDots = detect_object(
        "right_group", "./plantGroupDetection/splittedPhotos/right.png"
    )
    rightGroupResult = teachableMachinePython(model, class_names, rightGroupPath)

    if rightGroupWhiteDots > leftGroupWhiteDots:
        rightGroup = "Group 1"
        leftGroup = "Group 2"
    else:
        rightGroup = "Group 2"
        leftGroup = "Group 1"

    # rightModelResult
    print("done")
    return {
        "leftGroup": [leftGroup, leftGroupResult],
        "rightGroup": [rightGroup, leftGroupResult],
    }


@app.route("/getLeftPlantGroup", methods=["GET"])
def get_left_plant_group():
    png_file_path = "./plantGroupDetection/splittedPhotos/left.png"

    try:
        with open(png_file_path, "rb") as file:
            # Read the PNG file content
            png_content = file.read()

        # Encode the PNG file content to Base64
        base64_encoded_png = base64.b64encode(png_content).decode("utf-8")

        return str(base64_encoded_png)

    except IOError:
        # Handle file not found or read error
        return jsonify({"error": "Unable to read the PNG file."}), 500


@app.route("/getRightPlantGroup", methods=["GET"])
def get_right_plant_group():
    png_file_path = "./plantGroupDetection/splittedPhotos/right.png"

    try:
        with open(png_file_path, "rb") as file:
            # Read the PNG file content
            png_content = file.read()

        # Encode the PNG file content to Base64
        base64_encoded_png = base64.b64encode(png_content).decode("utf-8")

        return str(base64_encoded_png)

    except IOError:
        # Handle file not found or read error
        return jsonify({"error": "Unable to read the PNG file."}), 500


if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=5000)
