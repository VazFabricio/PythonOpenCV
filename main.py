import cv2
import numpy as np

window_width = 640
window_height = 480

# Define the color ranges for green, red, and blue (in HSV format)
lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])

lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

# Function to process each frame from the camera
def process_frame(frame):
    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(hsv_frame, (5, 5), 0)

    # Create masks for the specified color ranges
    green_mask = cv2.inRange(blurred, lower_green, upper_green)
    blue_mask = cv2.inRange(blurred, lower_blue, upper_blue)
    red_mask1 = cv2.inRange(blurred, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(blurred, lower_red2, upper_red2)

    # Combine the red masks
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Find contours of the colored objects in the masks
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around the detected objects
    for contour in green_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for contour in red_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    for contour in blue_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return frame


# Initialize the camera
cap = cv2.VideoCapture(0)

cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera', window_width, window_height)

while True:
    # Read the current frame from the camera
    ret, frame = cap.read()

    if not ret:
        break

    # Process the frame
    processed_frame = process_frame(frame)

    # Display the frame with the rectangles
    cv2.imshow('Camera', processed_frame)

    # Check for keypresses
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()