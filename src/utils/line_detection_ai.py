import cv2
import numpy as np
from ultralytics import YOLO
import json
import uuid
from pathlib import Path
from tqdm import tqdm  # Progress bar

# Predefined colors for each class
CLASS_COLORS = {
    0: (255, 0, 0),    # Red for Class 0
    1: (0, 255, 0),    # Green for Class 1
    2: (0, 0, 255),    # Blue for Class 2
    3: (255, 255, 0),  # Cyan for Class 3
    4: (255, 0, 255),  # Magenta for Class 4
    5: (0, 255, 255),  # Yellow for Class 5
    # Add more colors as needed for additional classes
}

# Default color if a class is not in the dictionary
DEFAULT_COLOR = (128, 128, 128)  # Gray

# Function to annotate all detected classes
def detect_lines(uploaded_file_path, results_dir, diagram_bbox, storage=None):
    # Load the image from the provided path
    print("Loading image...")
    image = cv2.imread(uploaded_file_path)
    
    # Preprocess the image for YOLO model
    print("Pre-processing image for YOLO model...")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
    dilated_image = cv2.dilate(blurred_image, np.ones((1, 1), np.uint8), iterations=1)
    pre_processed_img = cv2.cvtColor(dilated_image, cv2.COLOR_GRAY2BGR)

    # Load the YOLO model
    print("Loading YOLO model...")
    model = YOLO('models/intui_LDM_01.pt')  # Replace with your trained weights

    # Run predictions on the preprocessed image
    print("Running predictions...")
    results = model.predict(source=pre_processed_img, save=False)

    all_annotations = []  # To store annotations for all detected classes

    # Draw the diagram bounding box (green) on the image
    x_min, y_min, x_max, y_max = diagram_bbox
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 4)  # Green box

    # Process each prediction and annotate the image
    for result in tqdm(results, desc="Processing predictions"):
        for box in tqdm(result.boxes, desc="Processing boxes", leave=False):
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            label = int(box.cls)  # Class ID
            confidence = box.conf[0]  # Confidence score

            # Get color for the class
            color = CLASS_COLORS.get(label, DEFAULT_COLOR)

            # Annotate the bounding box
            class_name = f"Class {label}"  # Replace with your class mapping if available
            unique_id = str(uuid.uuid4())[:8]  # Generate a unique ID for the object
            annotation_text = f"{class_name} ({confidence:.2f})"

            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, annotation_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Add to annotations list
            all_annotations.append({
                "id": unique_id,
                "class": class_name,
                "confidence": round(float(confidence), 2),
                "bbox": [x1, y1, x2, y2]
            })

    # Save JSON result for all annotations
    json_result = {
        "detections": all_annotations
    }
    json_filename = Path(results_dir) / f"{Path(uploaded_file_path).stem}_detected_lines.json"
    with open(json_filename, 'w') as json_file:
        json.dump(json_result, json_file, indent=4)
    print(f"Saved JSON result to {json_filename}")

    # Annotated image save path
    annotated_image_path = Path(results_dir) / f"{Path(uploaded_file_path).stem}_detected_lines.jpg"
    cv2.imwrite(str(annotated_image_path), image)
    print(f"Saved annotated image to {annotated_image_path}")

    return str(annotated_image_path), str(json_filename)


# Entry point when running as a standalone script
if __name__ == '__main__':
    uploaded_file_path = 'samples/images/2.jpg'
    results_dir = 'results'
    diagram_bbox = [500, 400, 10000, 10000]  # Example bounding box
    storage = None  # Placeholder for storage logic

    # Call the detect_and_annotate function
    annotated_image_path, json_path = detect_lines(uploaded_file_path, results_dir, diagram_bbox, storage=storage)
    print(f"Annotated image saved at: {annotated_image_path}")
    print(f"JSON result saved at: {json_path}")
