from ultralytics import YOLO
import cv2
import numpy as np
import cvzone

# Load the trained YOLO model
model = YOLO("best.pt")
class_names = model.names  # Get class names from the model

# Load the image from file
img = cv2.imread('Pothole-006.jpg')
#return cv2.imdecode(np.fromfile(filename, np.uint8), flags)

# Check if the image was loaded properly
if img is None:
    print("Error: Image not found or cannot be opened.")
else:
    # Resize the image (optional, depending on model requirements)
    img = cv2.resize(img, (1020, 500))
    h, w, _ = img.shape  # Get image dimensions

    # Perform the YOLO model prediction on the image
    results = model.predict(img)

    # Process each detection result
    for r in results:
        boxes = r.boxes  # Get bounding boxes
        masks = r.masks  # Get segmentation masks if available

        # If masks are present, process them
        if masks is not None:
            masks = masks.data.cpu().numpy()  # Convert to NumPy array for OpenCV
            for seg, box in zip(masks, boxes):
                seg = cv2.resize(seg, (w, h))  # Resize the mask to match the image dimensions
                contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Draw contours and labels on the image
                for contour in contours:
                    d = int(box.cls)  # Get the class index
                    c = class_names[d]  # Get class name from index
                    x, y, width, height = cv2.boundingRect(contour)  # Get bounding box around the contour

                    # Draw the contour and label on the image
                    cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                    cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the image with detected potholes
    cv2.imshow('Pothole Detection', img)

    # Wait for a key press to close the image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

