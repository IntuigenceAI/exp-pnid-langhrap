# Relationship mapping module
import math

def calculate_distance(point1, point2):
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def map_relationships(symbols, text_annotations):
    """
    Maps relationships between symbols based on their proximity and text annotations.

    Args:
        symbols (list): List of classified symbols with positions.
        text_annotations (list): List of detected text annotations.

    Returns:
        list: List of relationships between symbols.
    """
    relationships = []
    for i, symbol1 in enumerate(symbols):
        for j, symbol2 in enumerate(symbols):
            if i >= j:  # Avoid duplicate pairs
                continue
            # Check proximity
            distance = calculate_distance(symbol1["position"], symbol2["position"])
            if distance < 100:  # Threshold for relationship proximity
                relationships.append({
                    "source": symbol1["id"],
                    "target": symbol2["id"],
                    "relationship": "connected"
                })

    # Annotate relationships with text if available
    for annotation in text_annotations:
        for relationship in relationships:
            if calculate_distance(annotation["position"], symbols[relationship["source"]]["position"]) < 50:
                relationship["annotation"] = annotation["text"]

    return relationships
