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
    from src.utils.parsing import parse_p_and_id
    from src.utils.classification import classify_symbols
    from src.utils.relationships import map_relationships
    from src.utils.graph_builder import build_knowledge_graph
    from src.utils.text_detection import process_image_for_text_detection, detect_texts
except ImportError as e:
    raise ImportError(f"Failed to import utility modules: {e}")

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the state schema
class AgentState(TypedDict):
    p_and_id_file: str
    extracted_symbols: Annotated[List[dict], operator.add]
    classified_components: Annotated[List[dict], operator.add]
    relationships: Annotated[List[dict], operator.add]
    knowledge_graph: dict

# Create the LangGraph workflow
workflow = StateGraph(AgentState)

try:
    # Add nodes to the workflow
    workflow.add_node("parse", parse_p_and_id)
    workflow.add_node("classify", classify_symbols)
    workflow.add_node("map_relationships", map_relationships)
    workflow.add_node("build_kg", build_knowledge_graph)

    # Define edges between nodes
    workflow.add_edge("parse", "classify")
    workflow.add_edge("classify", "map_relationships")
    workflow.add_edge("map_relationships", "build_kg")

    # Set entry and finish points
    workflow.set_entry_point("parse")
    workflow.set_finish_point("build_kg")

    # Compile the graph
    graph = workflow.compile()
    logger.info("Workflow successfully compiled!")
except Exception as e:
    logger.error(f"Error setting up the workflow: {e}")
    raise
