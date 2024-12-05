__all__ = ["process_image_for_text_detection", "extract_text", "detect_texts", "detect_text"]

import os
import json
import io
from PIL import Image, ImageDraw, ImageFont
from doctr.models import ocr_predictor
import numpy as np
import uuid  # For generating unique text IDs
import tensorflow as tf
from src.utils.storage import StorageInterface

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)  # Optimize memory growth
else:
    print("No GPU found. Using CPU instead.")


def process_image_for_text_detection(image_path: str, storage: StorageInterface):
    """
    Wrapper function for extracting text data from an image.
    """
    return extract_text(image_path, storage)


def extract_text(image_path: str, storage: StorageInterface):
    """
    Uses doctr to extract text data from an image.

    Parameters:
        image_path (str): Path to the input image.
        storage (StorageInterface): Storage interface for file operations.

    Returns:
        result (doctr document): Structured OCR result from doctr.
    """
    try:
        # Load doctr OCR model
        model = ocr_predictor(pretrained=True)

        # Load the image
        image_data = storage.load_file(image_path)
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)

        # Perform OCR on the image
        result = model([image_np])
        return result
    except Exception as e:
        raise RuntimeError(f"Error extracting text from image: {e}")


def detect_texts(image_path: str, result_path: str, diagram_bbox: list, storage: StorageInterface):
    """
    Detects texts from the P&ID image and saves annotated images and JSON with text information.

    Parameters:
        image_path (str): Path to the input image.
        result_path (str): Directory to save outputs.
        diagram_bbox (list): The bounding box of the diagram area to limit text detection.
        storage (StorageInterface): Storage interface for file operations.

    Outputs:
        Annotated image: '<input_file_name>_detected_texts.jpg'.
        JSON file: '<input_file_name>_detected_texts.json'.
    """
    try:
        # Load the image
        image_data = storage.load_file(image_path)
        image = Image.open(io.BytesIO(image_data))

        if image is None:
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Convert to RGB if the image is grayscale
        image = image.convert("RGB")

        # Extract text data using doctr
        result = extract_text(image_path, storage)

        # Create a drawing object to annotate the image
        draw = ImageDraw.Draw(image)

        # Set font for annotations
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        structured_texts = []  # List to store detected text blocks

        # Parse OCR result
        doc = result.export()
        for page in doc["pages"]:
            for block in page["blocks"]:
                for line in block["lines"]:
                    for word in line["words"]:
                        box = word["geometry"]
                        text = word["value"]
                        confidence = word.get("confidence", 1.0)

                        # Calculate bounding box coordinates
                        x_coords = [coord[0] for coord in box]
                        y_coords = [coord[1] for coord in box]
                        x_min = min(x_coords) * image.width
                        x_max = max(x_coords) * image.width
                        y_min = min(y_coords) * image.height
                        y_max = max(y_coords) * image.height
                        bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]

                        # Check if within diagram bounds
                        if is_within_diagram(bbox, diagram_bbox):
                            draw.rectangle(bbox, outline="green", width=2)
                            draw.text((x_min, y_min - 15), text, fill="red", font=font)
                            structured_texts.append(
                                {
                                    "text_id": str(uuid.uuid4()),
                                    "content": text,
                                    "confidence": confidence,
                                    "bbox": bbox,
                                }
                            )

        # Prepare file names
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_image_path = os.path.join(result_path, f"{image_name}_detected_texts.jpg")
        output_json_path = os.path.join(result_path, f"{image_name}_detected_texts.json")

        # Save outputs
        image_byte_array = io.BytesIO()
        image.save(image_byte_array, format="JPEG")
        storage.save_file(output_image_path, image_byte_array.getvalue())
        storage.save_file(output_json_path, json.dumps(structured_texts, indent=4).encode("utf-8"))

        return output_json_path, output_image_path

    except Exception as e:
        raise RuntimeError(f"Error detecting texts: {e}")


def detect_text(image_path: str, storage: StorageInterface):
    """
    Wrapper for detect_texts for simplified use.
    """
    # Default values for testing or single-detection use
    diagram_bbox = [0, 0, 10000, 10000]  # No bounding box restriction
    result_path = "results"
    return detect_texts(image_path, result_path, diagram_bbox, storage)


if __name__ == "__main__":
    from src.utils.storage import StorageFactory

    image_path = "samples/images/55A0171D02_pdf_p0001.png"
    result_path = "results"
    diagram_bbox = [100, 200, 300, 400]  # Example diagram bounding box

    storage = StorageFactory.get_storage()
    detect_texts(image_path, result_path, diagram_bbox, storage)
