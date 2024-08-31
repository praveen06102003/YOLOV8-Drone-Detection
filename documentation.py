import cv2
from ultralytics import YOLO

# Load your custom YOLOv8 model
model = YOLO("drone.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Initialize variables for object tracking
tracker = cv2.TrackerKCF_create()  # You can choose other trackers such as MIL, CSRT, etc.
initBB = None

# Function to check if the detected object is a drone
def is_drone(class_id):
    drone_class_id = 0  # Replace with actual class ID for drone
    return class_id == drone_class_id

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture frame")
        break

    if initBB is None:
        # Perform detection using YOLO
        results = model(frame)
        
        # Process detection results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = box.conf[0]
                
                print(f"Detected object with class_id: {class_id}, confidence: {confidence}")
                
                if is_drone(class_id) and confidence > CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    print(f"Drone detected with coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                    initBB = (x1, y1, x2 - x1, y2 - y1)
                    tracker.init(frame, initBB)
                    break
    
    if initBB is not None:
        # Update the tracker and get the updated position
        success, box = tracker.update(frame)
        
        if success:
            (x, y, w, h) = [int(v) for v in box]
            # Draw a red rectangle around the object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # Label the object as "Drone"
            cv2.putText(frame, "Drone", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # Display the coordinates above the rectangle
            coordinates_text = f"x1={x}, y1={y}, x2={x + w}, y2={y + h}"
            cv2.putText(frame, coordinates_text, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            print("Tracking failure detected. Resetting tracker.")
            initBB = None  # Reset tracker if tracking fails

    # Display the resulting frame
    cv2.imshow("Frame", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
