import cv2
from PIL import Image
import numpy as np

def get_limits(colors):
    limits = []
    for color in colors:
        c = np.uint8([[color]])
        hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

        lowerLimit = hsvC[0][0][0] - 10, 150, 100
        upperLimit = hsvC[0][0][0] + 10, 255, 255

        lowerLimit = np.array(lowerLimit, dtype=np.uint8)
        upperLimit = np.array(upperLimit, dtype=np.uint8)

        limits.append((lowerLimit, upperLimit))

    return limits

window_width = 720
window_height = 480

colors = [
    [60, 255, 50],  # Green
    [5, 150, 100],    # Red (lower range)
    [25, 255, 255], # Red (upper range)
    [100, 150, 50] # Blue
]

color_box = [
    (0, 255, 0),    # Green box color
    (0, 0, 255),    # Red box color
    (0, 0, 255),    # Red box color
    (255, 0, 0)     # Blue box color
]

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    limits = get_limits(colors)

    for i, color in enumerate(colors):
        lowerLimit, upperLimit = limits[i]
        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

        mask = Image.fromarray(mask)

        bbox = mask.getbbox()

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color_box[i], 5)

    frame = cv2.resize(frame, (window_width, window_height))

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
