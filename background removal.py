import cv2
import numpy as np

# Attach camera indexed as 0
camera = cv2.VideoCapture(0)

# Setting frame width and frame height as 640 x 480
camera.set(3, 640)
camera.set(4, 480)

# Loading the mountain image
mountain = cv2.imread('mount everest.jpg')

while True:
    # Read a frame from the attached camera
    status, frame = camera.read()

    # If we got the frame successfully
    if status:
        # Flip it
        frame = cv2.flip(frame, 1)

        # Converting the image to RGB for easy processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Creating thresholds for person detection (for simplicity, you can use pre-defined thresholds)
        lower_bound = np.array([100, 100, 100])  
        upper_bound = np.array([255, 255, 255])  

        # Thresholding image to detect person
        mask = cv2.inRange(frame_rgb, lower_bound, upper_bound)

        # Inverting the mask to get person's area
        mask = cv2.bitwise_not(mask)

        # Bitwise AND operation to extract person from the frame
        person = cv2.bitwise_and(frame, frame, mask=mask)

        # Replace the background with the mountain image
        mountain_resized = cv2.resize(mountain, (frame.shape[1], frame.shape[0]))  # Resize mountain to match frame size
        final_image = cv2.bitwise_and(mountain_resized, mountain_resized, mask=mask) + person

        # Show the final image
        cv2.imshow('Anywhere Selfie Booth', final_image)

        # Wait for 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code == 32:  # Break the loop when spacebar is pressed
            break

# Release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()
