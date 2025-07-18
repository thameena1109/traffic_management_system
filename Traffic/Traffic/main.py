import cv2
import numpy as np
import time
import os

# Paths for YOLOv3 model files
weights_path = r"C:\Users\Thameena\Desktop\sriram_traffic\yolo.weights"  
config_path = r"C:\Users\Thameena\Desktop\sriram_traffic\yolov3.cfg"       # Path to YOLOv3 config
coco_names = r"C:\Users\Thameena\Desktop\sriram_traffic\coco.names"        # Path to COCO names

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
video_path1 = r"C:\Users\Thameena\Desktop\sriram_traffic\V1.mp4"  # Path to your first video file
video_path2 = r"C:\Users\Thameena\Desktop\sriram_traffic\V2.mp4"  # Path to your second video file

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

# Function to calculate speed of the vehicle
def calculate_speed(box, current_frame):
    # Placeholder for object tracking mechanism
    frame_rate = video1.get(cv2.CAP_PROP_FPS)  # Adjust if needed
    distance_per_frame = 0.05  # Adjust this based on real-world measurements
    
    (startX, startY, endX, endY) = box
    box_width = endX - startX
    speed = (box_width / distance_per_frame) * frame_rate
    
    return speed

# Traffic signal regulation variables
green_signal_duration = 15  # Default green signal time
red_signal = [True, True]  # Initially both signals are red
last_switch_time = [time.time(), time.time()]
vehicle_counts_during_red = [[], []]  # Store vehicle counts during the red signal for both signals

def regulate_signals(vehicle_count1, vehicle_count2):
    global red_signal, last_switch_time, green_signal_duration

    current_time = time.time()

    if not red_signal[0] and current_time - last_switch_time[0] >= green_signal_duration:
        red_signal[0] = True
        red_signal[1] = False  # Switch to green for the second signal
        last_switch_time[1] = current_time
        calculate_green_duration(1)  # Calculate the next green signal duration for video 2

    elif not red_signal[1] and current_time - last_switch_time[1] >= green_signal_duration:
        red_signal[1] = True
        red_signal[0] = False  # Switch to green for the first signal
        last_switch_time[0] = current_time
        calculate_green_duration(0)  # Calculate the next green signal duration for video 1

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
    global green_signal_duration

    if len(vehicle_counts_during_red[signal_index]) > 0:
        average_vehicle_count = sum(vehicle_counts_during_red[signal_index]) / len(vehicle_counts_during_red[signal_index])
        
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

        vehicle_counts_during_red[signal_index].clear()

# Initialize timers
timer1 = 0
timer2 = 0
start_time1 = None
start_time2 = None

while True:
    # Capture frame-by-frame from both videos
    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()

    if not ret1 and not ret2:
        print("End of both videos.")
        break

    if not ret1:
        print("End of video 1.")
    if not ret2:
        print("End of video 2.")

    # Initialize vehicle counts
    vehicle_count1 = 0
    vehicle_count2 = 0

    # Process frame from the first video
    if ret1:
        (h1, w1) = frame1.shape[:2]
        blob1 = cv2.dnn.blobFromImage(frame1, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob1)
        detections1 = net.forward(output_layers)
        detected_vehicles1 = []
        detected_boxes1 = []  # To store bounding boxes

        for output in detections1:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.2 and CLASSES[classID] in ["car", "bus", "motorbike", "train"]:
                    box = detection[0:4] * np.array([w1, h1, w1, h1])
                    (centerX, centerY, width, height) = box.astype("int")
                    startX = int(centerX - (width / 2))
                    startY = int(centerY - (height / 2))

                    endX = startX + int(width)
                    endY = startY + int(height)

                    detected_boxes1.append((startX, startY, endX, endY))
                    label = "{}: {:.2f}%".format(CLASSES[classID], confidence * 100)
                    cv2.rectangle(frame1, (startX, startY), (endX, endY), COLORS[classID], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame1, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[classID], 2)

                    if is_new_detection((startX, startY, endX, endY), detected_vehicles1):
                        detected_vehicles1.append((startX, startY, endX, endY))

        vehicle_count1 = len(detected_vehicles1)

        # Calculate and display speed for detected vehicles
        for box in detected_boxes1:
            speed = calculate_speed(box, video1.get(cv2.CAP_PROP_POS_FRAMES))
            if speed is not None:
                label = f"Speed: {speed:.2f} m/s"
                (startX, startY, endX, endY) = box
                cv2.putText(frame1, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if red_signal[0]:
            vehicle_counts_during_red[0].append(vehicle_count1)

    # Process frame from the second video
    if ret2:
        (h2, w2) = frame2.shape[:2]
        blob2 = cv2.dnn.blobFromImage(frame2, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob2)
        detections2 = net.forward(output_layers)
        detected_vehicles2 = []
        detected_boxes2 = []  # To store bounding boxes

        for output in detections2:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.2 and CLASSES[classID] in ["car", "bus", "motorbike", "train"]:
                    box = detection[0:4] * np.array([w2, h2, w2, h2])
                    (centerX, centerY, width, height) = box.astype("int")
                    startX = int(centerX - (width / 2))
                    startY = int(centerY - (height / 2))

                    endX = startX + int(width)
                    endY = startY + int(height)

                    detected_boxes2.append((startX, startY, endX, endY))
                    label = "{}: {:.2f}%".format(CLASSES[classID], confidence * 100)
                    cv2.rectangle(frame2, (startX, startY), (endX, endY), COLORS[classID], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame2, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[classID], 2)

                    if is_new_detection((startX, startY, endX, endY), detected_vehicles2):
                        detected_vehicles2.append((startX, startY, endX, endY))

        vehicle_count2 = len(detected_vehicles2)

        # Calculate and display speed for detected vehicles
        for box in detected_boxes2:
            speed = calculate_speed(box, video2.get(cv2.CAP_PROP_POS_FRAMES))
            if speed is not None:
                label = f"Speed: {speed:.2f} m/s"
                (startX, startY, endX, endY) = box
                cv2.putText(frame2, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if red_signal[1]:
            vehicle_counts_during_red[1].append(vehicle_count2)

    # Regulate traffic signals
    regulate_signals(vehicle_count1, vehicle_count2)

    # Display the signal status
    signal_status1 = "GREEN" if not red_signal[0] else "RED"
    signal_status2 = "GREEN" if not red_signal[1] else "RED"

    if ret1:
        cv2.putText(frame1, f"Signal: {signal_status1}", (10, h1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if red_signal[0] else (0, 255, 0), 2)
        cv2.imshow("Video 1", frame1)

    if ret2:
        cv2.putText(frame2, f"Signal: {signal_status2}", (10, h2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if red_signal[1] else (0, 255, 0), 2)
        cv2.imshow("Video 2", frame2)

    # Break the loop if 'ESC' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources and close windows
video1.release()
video2.release()
cv2.destroyAllWindows()
