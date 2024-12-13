import sys
import os
import logging
from typing import TypedDict, Annotated, List
import operator
from langgraph.graph import StateGraph, END

# Add src to the Python path dynamically
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
if src_dir not in sys.path:
    sys.path.append(src_dir)


# Import utilities
try:
    from utils.parsing import parse_p_and_id
    from utils.classification import classify_symbols
    from utils.relationships import map_relationships
    from utils.graph_builder import build_knowledge_graph
    from utils.text_detection import process_image_for_text_detection, detect_texts
except ImportError as e:
    raise ImportError(f"Failed to import utility modules: {e}")
 

import json 
# Example usage
file_path = "/Users/hivamoh/Desktop/IntuigenceAI/exp-pnid-langhrap/src/utils/001.jpg"
result = parse_p_and_id(file_path)

print(result)
print(json.dumps(result, indent=2))

with open(result['symbols'][1], 'r', encoding='utf-8') as file:
    symbols = json.load(file)
# print(symbols)
symbols_classified = classify_symbols(symbols)
# print('FINAL')
print(symbols_classified)

with open(result['text_annotations'][0], 'r', encoding='utf-8') as file:
    text_annotations = json.load(file)
# print(text_annotations)
relationships = map_relationships(symbols_classified, text_annotations)
# print('FINAL')
# print(symbols_classified)
print(relationships)
graph_dict = build_knowledge_graph(symbols_classified, relationships)
print(graph_dict)


# # Enable logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Define the state schema
# class AgentState(TypedDict):
#     p_and_id_file: str
#     extracted_symbols: Annotated[List[dict], operator.add]
#     classified_components: Annotated[List[dict], operator.add]
#     relationships: Annotated[List[dict], operator.add]
#     knowledge_graph: dict

# # Create the LangGraph workflow
# workflow = StateGraph(AgentState)

# try:
#     # Add nodes to the workflow
#     workflow.add_node("parse", parse_p_and_id)
#     workflow.add_node("classify", classify_symbols)
#     workflow.add_node("map_relationships", map_relationships)
#     workflow.add_node("build_kg", build_knowledge_graph)

#     # Define edges between nodes
#     workflow.add_edge("parse", "classify")
#     workflow.add_edge("classify", "map_relationships")
#     workflow.add_edge("map_relationships", "build_kg")

#     # Set entry and finish points
#     workflow.set_entry_point("parse")
#     workflow.set_finish_point("build_kg")

#     # Compile the graph
#     graph = workflow.compile()
#     logger.info("Workflow successfully compiled!")
#     from IPython.display import Image
#     output_image_path = "workflow_diagram.png"
#     # draw_mermaid_png() should return the PNG image as binary data
#     png_data = graph.get_graph().draw_mermaid_png()

#     # Write the binary data to a file
#     with open(output_image_path, "wb") as f:
#         f.write(png_data)
    
    
# except Exception as e:
#     logger.error(f"Error setting up the workflow: {e}")
#     raise
