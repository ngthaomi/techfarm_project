import cv2
import numpy as np

def stretch_contrast(image):
    # Normalize the image to the range [0, 1]
    normalized_image = image.astype(np.float32) / 255.0

    # Apply contrast stretching
    stretched_image = (normalized_image - np.min(normalized_image)) / (np.max(normalized_image) - np.min(normalized_image))

    # Scale the image back to the range [0, 255]
    stretched_image = (stretched_image * 255).astype(np.uint8)

    return stretched_image

def binarize_image(image):
    # Perform binary thresholding
    _, binary_image = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)

    return binary_image

def reduce_noise(binary_image, kernel_size=3, iterations=1):
    # Define the kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Apply morphological opening to reduce noise
    opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations)

    return opened_image

def detect_object(group_name, image_path):
    # Load the input image
    image = cv2.imread(image_path)

    # Extract the red channel
    red_channel = image[:, :, 2]

    # Save the red channel as PNG with red color
    red_channel_path = group_name + '_red_channel.png'
    red_channel_color = cv2.cvtColor(red_channel, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(red_channel_path, red_channel_color)
    # print(f"Red channel saved at {red_channel_path}")

    # Apply contrast stretching to the red channel
    stretched_red = stretch_contrast(red_channel)

    # Save the stretched red channel as PNG
    stretched_red_path = group_name + '_stretched_red.png'
    cv2.imwrite(stretched_red_path, stretched_red)
    # print(f"Stretched red channel saved at {stretched_red_path}")

    # Binarize the stretched red channel
    binary_image = binarize_image(stretched_red)

    # Save the binary image as PNG
    binary_image_path = group_name + '_binary_image.png'
    cv2.imwrite(binary_image_path, binary_image)
    # print(f"Binary image saved at {binary_image_path}")

    # Reduce noise in binary image
    denoised_image = reduce_noise(binary_image, kernel_size=3, iterations=1)

    # Save the denoised image as PNG
    denoised_image_path = group_name + '_denoised_image.png'
    cv2.imwrite(denoised_image_path, denoised_image)
    # print(f"Denoised image saved at {denoised_image_path}")

    # Find contours in the denoised image
    contours, _ = cv2.findContours(denoised_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count the number of white dots (contours)
    white_dot_count = len(contours)

    # # print the number of white dots
    # print(f"Number of white dots: {white_dot_count}")

    # Get the coordinates of the top-leftmost white dot
    top_left_dot_coordinates = None
    if white_dot_count > 0:
        (x, y, w, h) = cv2.boundingRect(contours[0])
        top_left_dot_coordinates = (x, y)

        # Draw a blue square around the top-leftmost white dot
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)

    # Save the image with the blue square
    output_path = group_name + '_output_image.png'
    cv2.imwrite(output_path, image)

    # # Return the coordinates of the top-leftmost white dot
    # return top_left_dot_coordinates

    # # print(f"Image with blue square saved at {output_path}")
    if white_dot_count == 0:
        print("None Found")
    elif white_dot_count < 10:
        print("Diamond Group 1")
        return "Group 1"
    elif white_dot_count >=  10:
        print("heart Group 2")
        return "Group 2"
        
def split_image(image_path):
    # Load the input image
    image = cv2.imread(image_path)

    # Get the width and height of the image
    height, width = image.shape[:2]

    # Calculate the middle point of the image
    middle_point = width // 2

    # Split the image into left and right halves
    left_image = image[:, :middle_point, :]
    right_image = image[:, middle_point:, :]

    # Save the left and right halves as PNG files
    cv2.imwrite("./splittedPhotos/left.png", left_image)
    cv2.imwrite("./splittedPhotos/right.png", right_image)


# # # Example usage
detect_object("group1", "./plantGroupDetection/plantGroupTesting/diamond1.jpg")
detect_object("group1", "./plantGroupDetection/plantGroupTesting/diamond2.jpg")
detect_object("group1", "./plantGroupDetection/plantGroupTesting/diamond3.jpg")

detect_object("group2", "./plantGroupDetection/plantGroupTesting/heart1.jpg")
detect_object("group2", "./plantGroupDetection/plantGroupTesting/heart2.jpg")
detect_object("group2", "./plantGroupDetection/plantGroupTesting/heart3.jpg")

