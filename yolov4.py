import cv2 as cv
import time
from collections import deque
import numpy as np
from scipy.signal import find_peaks

def detect_cars(video_file):
    try:
        # Set thresholds
        Conf_threshold = 0.4
        NMS_threshold = 0.4

        # Load class names from file
        class_name = []
        with open('classes.txt', 'r') as f:
            class_name = [cname.strip() for cname in f.readlines()]

        if "car" not in class_name:
            raise ValueError("Class 'car' not found in classes.txt")

        # Load the YOLO model
        net = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

        # Check if CUDA is available, otherwise use CPU
        try:
            net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
        except:
            net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
            net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        # Initialize the detection model
        model = cv.dnn_DetectionModel(net)
        model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

        # Open the video file
        cap = cv.VideoCapture(video_file)
        if not cap.isOpened():
            raise FileNotFoundError(f"Error: Cannot open video file {video_file}")

        starting_time = time.time()
        frame_counter = 0

        # To keep track of car counts over time
        car_counts = deque()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1

            # Perform detection
            classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)

            # Count the number of cars detected
            car_count = sum(1 for classid in classes if class_name[classid] == "car")

            # Record the car count with the current timestamp
            current_time = time.time()
            car_counts.append((current_time, car_count))

            # Remove counts older than 30 seconds
            while car_counts and car_counts[0][0] < current_time - 30:
                car_counts.popleft()

        # Extract the car counts from the deque
        car_count_values = [count for _, count in car_counts]

        # Find peaks in the car count values
        peaks, _ = find_peaks(car_count_values)

        # Calculate the mean of the peak values
        mean_peak_value = np.mean([car_count_values[i] for i in peaks]) if peaks.size > 0 else 0

        # Release resources
        cap.release()

        return mean_peak_value

    except Exception as e:
        print(f"Error in detect_cars: {e}")
        return -1  # Return -1 to indicate an error

