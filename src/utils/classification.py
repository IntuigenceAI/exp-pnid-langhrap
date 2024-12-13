# Classification module
import json

# Example mapping of symbol types to their classifications
SYMBOL_CLASSIFICATION_MAP = {
    "valve": ["gate_valve", "ball_valve"],
    "tank": ["storage_tank", "mixing_tank"],
    "pipeline": ["pipe", "connection"],
    "sensor": ["temperature_sensor", "pressure_sensor"]
}

def classify_symbols(symbols):

    """
    Classifies symbols based on predefined categories.
    
    Args:
        symbols (list): List of detected symbols with types.

    Returns:
        list: List of symbols with classifications added.
    """
    classified_symbols = []
    for symbol in symbols:
        symbol_type = symbol.get("type", "unknown")
        # print(symbol_type)
        classification = "unknown"
        for category, types in SYMBOL_CLASSIFICATION_MAP.items():
            if symbol_type in types:
                classification = category
                break
        symbol["classification"] = classification
        classified_symbols.append(symbol)
    return classified_symbols



