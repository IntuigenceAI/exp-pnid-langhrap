# detect_symbols.py

import cv2
import json
import uuid
import os
from ultralytics import YOLO
from tqdm import tqdm
from src.utils.storage import StorageInterface
import numpy as np

def detect_symbols(image_path, results_dir="results", file_name="", apply_preprocessing=False, storage=None):
    """
    Wrapper function for symbol detection using run_detection_with_optimal_threshold.
    """
    return run_detection_with_optimal_threshold(
        image_path=image_path,
        results_dir=results_dir,
        file_name=file_name,
        apply_preprocessing=apply_preprocessing,
        storage=storage
    )


# Preprocessing for symbol detection
def preprocess_image_for_symbol_detection(image_cv):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    filtered = cv2.bilateralFilter(equalized, 9, 75, 75)
    edges = cv2.Canny(filtered, 100, 200)
    preprocessed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return preprocessed_image

# Function to evaluate the quality of detections (based on your own logic)
def evaluate_detections(detections_list):
    return len(detections_list)  # Simple metric: count of detections

# Detection function with automatic confidence threshold selection
def run_detection_with_optimal_threshold(image_path, results_dir="results", file_name="", apply_preprocessing=False, storage: StorageInterface = None):
    # Load the image using the storage interface
    image_data = storage.load_file(image_path)
    nparr = np.frombuffer(image_data, np.uint8)
    image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if apply_preprocessing:
        print("Preprocessing image for symbol detection...")
        image_cv = preprocess_image_for_symbol_detection(image_cv)
    else:
        print("Skipping image preprocessing for symbol detection...")

    # Get the latest model path for symbol detection
    model_path = "models/Intui_SDM_15.pt"
    if not model_path:
        return "Error: No model found!", None, None, None, None, None, None, None

    model = YOLO(model_path)

    best_confidence_threshold = 0.5  # Default value
    best_detections_list = []
    best_metric = -1

    # Iterate over different confidence thresholds
    for confidence_threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(f"Running detection with confidence threshold: {confidence_threshold}...")
        results = model.predict(source=image_cv, imgsz=1280)

        detections_list = []
        for result in tqdm(results, desc="Processing results"):
            for box in tqdm(result.boxes, desc="Processing boxes", leave=False):
                confidence = float(box.conf[0])
                if confidence >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    label = result.names[class_id]

                    # Parse label (category and type)
                    split_label = label.split('_')
                    if len(split_label) >= 2:
                        category = split_label[0]
                        type_ = split_label[1]
                        new_label = ' '.join(split_label[2:])
                    else:
                        print(f"Unexpected label format: {label}. Skipping this detection.")
                        continue

                    detection_id = str(uuid.uuid4())
                    detection_info = {
                        "symbol_id": detection_id,
                        "class_id": class_id,
                        "category": category,
                        "type": type_,
                        "label": new_label,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2]
                    }
                    detections_list.append(detection_info)

        # Evaluate detection quality
        metric = evaluate_detections(detections_list)

        if metric > best_metric:
            best_metric = metric
            best_confidence_threshold = confidence_threshold
            best_detections_list = detections_list

    print(f"Best confidence threshold selected: {best_confidence_threshold}")

    # Annotating detections
    for det in best_detections_list:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        confidence = det["confidence"] * 100  # Convert to percentage
        annotation_text = f'{label}\nConf: {confidence:.0f}%'
        text_color = (0, 0, 255)
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)
        y_text = y1 - 20 if y1 - 20 > 20 else y1 + 20
        for i, line in enumerate(annotation_text.split('\n')):
            cv2.putText(image_cv, line, (x1, y_text + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    # Saving detection results
    storage.create_directory(results_dir)
    
    # Fixing file name to remove extension before appending details
    file_name_without_extension = os.path.splitext(file_name)[0]
    
    detection_json_path = os.path.join(results_dir, f'{file_name_without_extension}_detected_symbols.json')
    storage.save_file(detection_json_path, json.dumps(best_detections_list, indent=4).encode('utf-8'))

    detection_image_path = os.path.join(results_dir, f'{file_name_without_extension}_detected_symbols.jpg')
    _, img_encoded = cv2.imencode('.jpg', image_cv)
    storage.save_file(detection_image_path, img_encoded.tobytes())

    # Bounding box
    diagram_bbox = [min([det['bbox'][0] for det in best_detections_list], default=0),
                    min([det['bbox'][1] for det in best_detections_list], default=0),
                    max([det['bbox'][2] for det in best_detections_list], default=0),
                    max([det['bbox'][3] for det in best_detections_list], default=0)]

    return detection_image_path, detection_json_path, f"Selected confidence threshold: {best_confidence_threshold}", diagram_bbox

if __name__ == "__main__":
    from storage import StorageFactory

    uploaded_file_path = "samples/images/01fig07_alt.jpg"
    results_dir = "results"
    apply_symbol_preprocessing = False  # Set to False if preprocessing isn't needed

    storage = StorageFactory.get_storage()

    detection_image_path, detection_json_path, detection_log_message, diagram_bbox = run_detection_with_optimal_threshold(
        uploaded_file_path,
        results_dir=results_dir,
        file_name=os.path.basename(uploaded_file_path),
        apply_preprocessing=apply_symbol_preprocessing,
        storage=storage
    )

    print("Detection Image Path:", detection_image_path)
    print("Detection JSON Path:", detection_json_path)
    print("Detection Log Message:", detection_log_message)
    print("Diagram BBox:", diagram_bbox)
