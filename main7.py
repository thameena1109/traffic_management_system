import cv2
import time
import torch
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [2, 3, 5, 7]  # Only vehicles: car, motorcycle, bus, truck

# Load 2 video feeds
video_paths = ['V1.mp4', 'V2.mp4']
caps = [cv2.VideoCapture(path) for path in video_paths]

def get_vehicle_count_and_detections(frame):
    results = model(frame)
    count = len(results.xyxy[0])
    return count, results

def draw_detections(frame, results, signal_status, vehicle_count=None, timer_text=None, timer_color=(255,255,255)):
    for *box, conf, cls in results.xyxy[0]:
        label = model.names[int(cls)]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    color = (0, 0, 255) if signal_status == "Red" else (0, 255, 0)
    cv2.putText(frame, f"Signal: {signal_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Draw vehicle count at top-left corner
    if vehicle_count is not None:
        cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw timer at top-right corner
    if timer_text is not None:
        text_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = frame.shape[1] - text_size[0] - 10
        cv2.putText(frame, timer_text, (text_x, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, timer_color, 2)

    return frame

def combine_1x2(frames):
    return np.hstack((frames[0], frames[1]))

def initialize_red_signal(duration=10):
    print(f"[INIT] Red Signal Phase: {duration} seconds")
    vehicle_counts = [0, 0]
    frame_counts = [0, 0]
    start_time = time.time()

    while time.time() - start_time < duration:
        combined_frames = []
        for i in range(2):
            ret, frame = caps[i].read()
            if not ret:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            count, results = get_vehicle_count_and_detections(frame)
            vehicle_counts[i] += count
            frame_counts[i] += 1
            frame = draw_detections(frame, results, "Red", vehicle_count=count)
            combined_frames.append(cv2.resize(frame, (640, 360)))

        matrix = combine_1x2(combined_frames)
        cv2.imshow("Traffic Monitoring [Red Phase]", matrix)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None
    average_counts = [vehicle_counts[i] / frame_counts[i] if frame_counts[i] > 0 else 0 for i in range(2)]
    return average_counts

def get_green_duration(count):
    if count <= 10:
        return 10
    elif count <= 20:
        return 20
    else:
        return 30

def green_signal_phase(green_idx, red_idx, green_duration):
    print(f"[GREEN] Green for Video {green_idx+1} - Duration: {green_duration} seconds")
    start_time = time.time()
    ret_red, frame_red = caps[red_idx].read()
    if not ret_red:
        frame_red = np.zeros((480, 640, 3), dtype=np.uint8)
    # Freeze red frame
    frozen_red_frame = cv2.resize(frame_red, (640, 360))

    while time.time() - start_time < green_duration:
        combined_frames = []
        # Green video plays and detects
        ret_green, frame_green = caps[green_idx].read()
        if not ret_green:
            frame_green = np.zeros((480, 640, 3), dtype=np.uint8)
        count_green, results_green = get_vehicle_count_and_detections(frame_green)

        # Calculate remaining time for timer
        elapsed = int(time.time() - start_time)
        remaining = max(0, int(green_duration - elapsed))
        timer_text = f"Timer: {remaining}"
        timer_color = (0, 255, 0)  # Green color for timer

        frame_green = draw_detections(frame_green, results_green, "Green", vehicle_count=count_green, timer_text=timer_text, timer_color=timer_color)
        combined_frames.append(cv2.resize(frame_green, (640, 360)))

        # Red video frozen frame with red signal and vehicle count
        frame_red_signal = frozen_red_frame.copy()
        cv2.putText(frame_red_signal, "Signal: Red", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # We do not have vehicle count for red frozen frame here, so no vehicle count or timer on red frozen frame
        combined_frames.append(frame_red_signal)

        matrix = combine_1x2(combined_frames)
        cv2.imshow("Traffic Monitoring [Green Phase]", matrix)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

def run_traffic_signal_cycle():
    # Red phase initialization
    vehicle_counts = initialize_red_signal()
    if vehicle_counts is None:
        return

    print(f"Average vehicle counts during red phase: {vehicle_counts}")

    # Determine which video has fewer vehicles to get green first
    if vehicle_counts[0] < vehicle_counts[1]:
        first_green = 0
        second_green = 1
    else:
        first_green = 1
        second_green = 0

    # Get green durations based on counts
    green_duration_first = get_green_duration(vehicle_counts[first_green])
    green_duration_second = get_green_duration(vehicle_counts[second_green])

    # First green phase
    green_signal_phase(first_green, second_green, green_duration_first)
    # Second green phase
    green_signal_phase(second_green, first_green, green_duration_second)

# Start system
run_traffic_signal_cycle()

# Cleanup
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
