import cv2
import time
import numpy as np

# Load YOLOv3 model
def load_yolo_model():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Adjust paths as necessary
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

# Process frames through YOLOv3 to detect vehicles
def process_frame(frame, net, output_layers):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filter for vehicles: class_id 2 = car, 5 = bus, 7 = truck
            if confidence > 0.5 and class_id in [2, 5, 7]:  # Detect only car, bus, and truck
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = box
        label = str(class_ids[i])
        color = (0, 255, 0)  # Green box for vehicles
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame, len(indices)  # Returning modified frame and number of detections

# Load YOLOv3 model
print("Loading YOLOv3 model...")
net, output_layers = load_yolo_model()
print("YOLOv3 model loaded successfully.")

# Initialize video captures
cap1 = cv2.VideoCapture('V1.mp4')  # Path to first video
cap2 = cv2.VideoCapture('V2.mp4')  # Path to second video

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: One or both videos failed to open.")
    exit()

# Start with 10 seconds of red light for both videos
print("Red light for 10 seconds...")
time.sleep(10)  # 10 second delay to simulate red light

# Main loop
while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # Check if either of the videos ended
    if not ret1 or not ret2:
        print("End of one or both videos.")
        break

    # Process both frames (vehicle detection)
    if ret1:
        frame1, count1 = process_frame(frame1, net, output_layers)  # Process frame1 through YOLOv3
    if ret2:
        frame2, count2 = process_frame(frame2, net, output_layers)  # Process frame2 through YOLOv3

    # Display the frames side by side
    if ret1 and ret2:
        combined_frame = cv2.hconcat([frame1, frame2])
        cv2.imshow('Video', combined_frame)

    # Press 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Terminating...")
        break

# Release resources
cap1.release()
cap2.release()
cv2.destroyAllWindows()
