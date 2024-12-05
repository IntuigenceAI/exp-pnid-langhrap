# Import necessary libraries and the detection modules
import cv2
import json
from PIL import Image
from src.utils.symbol_detection import preprocess_image_for_symbol_detection, detect_symbols
from src.utils.text_detection import process_image_for_text_detection, detect_text

def parse_p_and_id(file_path):
    """
    Parses a P&ID file to extract symbols and text annotations.
    
    Args:
        file_path (str): Path to the input P&ID image file.

    Returns:
        dict: Structured data containing detected symbols and text.
    """
    # Load the image
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Unable to load image: {file_path}")

    # Step 1: Detect Symbols
    print("Detecting symbols...")
    preprocessed_image = preprocess_image_for_symbol_detection(image)
    symbols = detect_symbols(preprocessed_image)
    
    # Step 2: Detect Text
    print("Detecting text...")
    image_pil = Image.open(file_path)
    text_annotations = detect_text(image_pil)
    
    # Combine results
    parsed_data = {
        "symbols": symbols,
        "text_annotations": text_annotations
    }
    return parsed_data

# Example usage
# file_path = "example_pid.png"
# result = parse_p_and_id(file_path)
# print(json.dumps(result, indent=2))
