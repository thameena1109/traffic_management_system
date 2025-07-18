import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Paths for YOLOv3 model files
weights_path = "C:/Users/Thameena/Desktop/tms/yolo.weights"  # Path to YOLOv3 weights
config_path = "C:/Users/Thameena/Desktop/tms/yolov3.cfg"       # Path to YOLOv3 config
coco_names = "C:/Users/Thameena/Desktop/tms/coco.names"        # Path to COCO names

# Function to check file existence
def check_file_existence(file_path, file_description):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: {file_description} file not found at {file_path}")

# Check if the required files exist
try:
    check_file_existence(weights_path, "YOLO weights")
    check_file_existence(config_path, "YOLO config")
    check_file_existence(coco_names, "COCO class labels")
except FileNotFoundError as e:
    print(e)
    exit()

# Load YOLO class labels from file
with open(coco_names, "r") as f:
    CLASSES = [line.strip() for line in f.readlines()]

# Random colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load YOLO model
print("Loading YOLOv3 model...")
try:
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    print("YOLOv3 model loaded successfully.")
except cv2.error as e:
    print(f"OpenCV error: {e}")
    exit()
except Exception as e:
    print(f"Error: {e}")
    exit()

# Initialize video capture for two video files
video_path1 = "C:/Users/Thameena/Desktop/tms/V1.mp4" # Replace with your first video file path
video_path2 = "C:/Users/Thameena/Desktop/tms/V2.mp4"  # Replace with your second video file path

video1 = cv2.VideoCapture(video_path1)
video2 = cv2.VideoCapture(video_path2)

if not video1.isOpened():
    print("Error: Could not open the first video file.")
    exit()

if not video2.isOpened():
    print("Error: Could not open the second video file.")
    exit()

print("Starting video processing...")

# Get output layer names from YOLO
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Function to check if a bounding box is new
def is_new_detection(box, detected_boxes):
    for dbox in detected_boxes:
        if (abs(dbox[0] - box[0]) < 50 and
            abs(dbox[1] - box[1]) < 50 and
            abs(dbox[2] - box[2]) < 50 and
            abs(dbox[3] - box[3]) < 50):
            return False
    return True

# Traffic signal regulation variables
green_signal_duration = 15  # Default green signal time
red_signal = [True, True]  # Initially both signals are red
last_switch_time = [time.time(), time.time()]
vehicle_counts_during_red = [[], []]  # Store vehicle counts during the red signal for both signals

def regulate_signals(vehicle_count1, vehicle_count2):
    global red_signal, last_switch_time, green_signal_duration

    current_time = time.time()

    # If the first signal is green and it's time to switch
    if not red_signal[0] and current_time - last_switch_time[0] >= green_signal_duration:
        red_signal[0] = True
        red_signal[1] = False  # Switch to green for the second signal
        last_switch_time[1] = current_time
        calculate_green_duration(1)  # Calculate the next green signal duration for video 2

    # If the second signal is green and it's time to switch
    elif not red_signal[1] and current_time - last_switch_time[1] >= green_signal_duration:
        red_signal[1] = True
        red_signal[0] = False  # Switch to green for the first signal
        last_switch_time[0] = current_time
        calculate_green_duration(0)  # Calculate the next green signal duration for video 1

    # If both signals are red, make the one with more vehicles green
    if red_signal[0] and red_signal[1]:
        if vehicle_count1 > vehicle_count2:
            red_signal[0] = False  # Green for the first
            last_switch_time[0] = current_time
            calculate_green_duration(0)  # Calculate the next green signal duration for video 1
        else:
            red_signal[1] = False  # Green for the second
            last_switch_time[1] = current_time
            calculate_green_duration(1)  # Calculate the next green signal duration for video 2

def calculate_green_duration(signal_index):
    """
    Calculate the green signal duration based on the average vehicle count during the red signal.
    """
    global green_signal_duration

    if len(vehicle_counts_during_red[signal_index]) > 0:
        average_vehicle_count = sum(vehicle_counts_during_red[signal_index]) / len(vehicle_counts_during_red[signal_index])
        
        # Set green signal duration based on the average vehicle count
        if 5 <= average_vehicle_count <= 10:
            green_signal_duration = 15
        elif 11 <= average_vehicle_count <= 15:
            green_signal_duration = 25
        elif 16 <= average_vehicle_count <= 20:
            green_signal_duration = 35
        elif 21 <= average_vehicle_count <= 30:
            green_signal_duration = 40
        elif average_vehicle_count >= 31:
            green_signal_duration = 60
        else:
            green_signal_duration = 15  # Default if fewer than 5 vehicles on average

        # Reset the count list after calculating the duration
        vehicle_counts_during_red[signal_index].clear()

# Initialize timers
start_time = time.time()
initial_red_signal_duration = 10

# Function to detect vehicles in a frame
def detect_vehicles(frame, net, output_layers):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)
    
    vehicle_count = 0
    boxes = []
    confidences = []
    class_ids = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Detect vehicles (cars, buses, trucks, motorbikes)
            if confidence > 0.5 and CLASSES[class_id] in ['car', 'bus', 'truck', 'motorbike']:
                # Extract bounding box coordinates
                box = obj[0:4] * np.array([width, height, width, height])
                (centerX, centerY, box_width, box_height) = box.astype("int")
                x = int(centerX - (box_width / 2))
                y = int(centerY - (box_height / 2))
                
                # Append the box, confidence, and class ID for NMS
                boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
                # Count the detected vehicle
                vehicle_count += 1

    # Apply non-maxima suppression to remove overlapping boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(idxs) > 0:
        for i in idxs.flatten():
            box = boxes[i]
            x, y, w, h = box
            color = [int(c) for c in COLORS[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{CLASSES[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return vehicle_count

# Function to update and display video frames using matplotlib
def update_frame(frame, ax, vehicle_count, average_vehicle_count, signal_color):
    ax.clear()
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Vehicle Count: {vehicle_count}, Avg Count: {average_vehicle_count:.2f}")
    
    # Draw traffic signal
    (h, w) = frame.shape[:2]
    if signal_color == 'red':
        cv2.rectangle(frame, (10, 10), (110, 60), (0, 0, 255), -1)  # Red
    elif signal_color == 'green':
        cv2.rectangle(frame, (10, 10), (110, 60), (0, 255, 0), -1)  # Green
    
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.pause(0.01)

# Create figures for displaying frames
fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))

# Process videos frame by frame
while True:
    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()

    if not ret1 and not ret2:
        break

    # Initialize vehicle counts
    vehicle_count1 = 0
    vehicle_count2 = 0

    # Process the first video frame
    if ret1:
        vehicle_count1 = detect_vehicles(frame1, net, output_layers)

    # Process the second video frame
    if ret2:
        vehicle_count2 = detect_vehicles(frame2, net, output_layers)

    # Update traffic signals
    if time.time() - start_time < initial_red_signal_duration:
        signal_color1 = 'red'
        signal_color2 = 'red'
    else:
        if red_signal[0]:
            signal_color1 = 'red'
        else:
            signal_color1 = 'green'

        if red_signal[1]:
            signal_color2 = 'red'
        else:
            signal_color2 = 'green'

        # Update signals and vehicle counts
        regulate_signals(vehicle_count1, vehicle_count2)
        if red_signal[0]:
            vehicle_counts_during_red[0].append(vehicle_count1)
        if red_signal[1]:
            vehicle_counts_during_red[1].append(vehicle_count2)

    # Display video frames using matplotlib
    if ret1:
        update_frame(frame1, ax1, vehicle_count1, np.mean(vehicle_counts_during_red[0]) if vehicle_counts_during_red[0] else 0, signal_color1)

    if ret2:
        update_frame(frame2, ax2, vehicle_count2, np.mean(vehicle_counts_during_red[1]) if vehicle_counts_during_red[1] else 0, signal_color2)

# Release video capture objects
video1.release()
video2.release()
cv2.destroyAllWindows()
plt.close(fig1)
plt.close(fig2)
