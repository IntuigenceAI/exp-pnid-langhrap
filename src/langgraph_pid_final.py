#####-----------------****** IMPORTS ******-----------------#####
import sys
import getpass
import os
import logging
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from typing import Annotated, List, Dict, Any, Sequence
import operator
import json

from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.agents import AgentAction
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter

#####-----------------****** ENVIRONMENT SETUP ******-----------------#####

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("LANGCHAIN_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangGraph_01"


from langchain_openai import ChatOpenAI

# Bind the tools to the model
model = ChatOpenAI(temperature=0, streaming=True)
# model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0, streaming=True)

# Add src to the Python path dynamically
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
if src_dir not in sys.path:
    sys.path.append(src_dir)
####-----------------****** TOOLS ******-----------------#####
# Import utilities
try:
    from utils.parsing import parse_p_and_id
    from utils.classification import classify_symbols
    from utils.relationships import map_relationships
    from utils.graph_builder import build_knowledge_graph
    from utils.text_detection import process_image_for_text_detection, detect_texts
except ImportError as e:
    raise ImportError(f"Failed to import utility modules: {e}")
 


@tool("symbol_detector", return_direct=True)
def extract_symbols_t(input:str) -> List[Dict[str, Any]]:
    """TOOL: Calls the CV Function to extract the symbols in the P&ID image"""
    # NOTE: This currently points to the json file, adjust later
    # with open("/Users/hivamoh/Desktop/IntuigenceAI/2/2_detected_symbols.json", 'r') as file:
    #     data = json.load(file)
    # return data["detections"]
    result = parse_p_and_id(input)
    return result
    
@tool("text_detector", return_direct=True)
def extract_texts_t(input:str) -> List[Dict[str, Any]]:
    """TOOL: Calls the CV Function to extract the text in the P&ID image"""
    # NOTE: This currently points to the json file, adjust later
    with open("/Users/hivamoh/Desktop/IntuigenceAI/2/2_detected_texts.json", 'r') as file:
        data = json.load(file)
    return data["detections"]


@tool("line_detector", return_direct=True)
def extract_lines_t(input:str) -> List[Dict[str, Any]]:
    """TOOL: Calls the CV Function to extract the lines in the P&ID image"""
    # NOTE: This currently points to the json file, adjust later
    with open("/Users/hivamoh/Desktop/IntuigenceAI/2/2_detected_lines.json", 'r') as file:
        data = json.load(file)
    return data["detections"]



import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool("document_writer", return_direct=True)
def write_document_t(processed_data: Dict[str, Any]) -> str:
    """
    TOOL: Writes the processed symbols, texts, and lines into a document.
    """
    try:
        document_path = "/Users/hivamoh/Desktop/IntuigenceAI/2/processed_p&ID_document.json"
        with open(document_path, 'w') as doc_file:
            json.dump(processed_data, doc_file, indent=4)
        logger.info(f"Document written successfully to {document_path}")
        return f"Document written successfully to {document_path}"
    except Exception as e:
        logger.error(f"Failed to write document: {e}")
        return f"Failed to write document: {e}"


#####-----------------****** REFLECTION AND CRITIQUE TOOLS ******-----------------#####
# Initialize a separate model for reflection, critique and check to ensure independence
reflection_model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0.5)
critique_model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0.5)
check_model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0.5)

@tool("symbol_reflector", return_direct=True)
def reflect_symbols(output_json: str) -> str:
    """
    REFLECTION TOOL: Evaluates the quality of the symbols extracted.
    """
    prompt = f"Evaluate the following extracted symbols for accuracy and completeness:\n{output_json}\nProvide feedback."
    reflection = reflection_model.invoke(prompt)
    return reflection

@tool("text_reflector", return_direct=True)
def reflect_texts(output_json: str) -> str:
    """
    REFLECTION TOOL: Evaluates the quality of the texts extracted.
    """
    prompt = f"Evaluate the following extracted texts for accuracy and completeness:\n{output_json}\nProvide feedback."
    reflection = reflection_model.invoke(prompt)
    return reflection

@tool("line_reflector", return_direct=True)
def reflect_lines(output_json: str) -> str:
    """
    REFLECTION TOOL: Evaluates the quality of the lines extracted.
    """
    prompt = f"Evaluate the following extracted lines for accuracy and completeness:\n{output_json}\nProvide feedback."
    reflection = reflection_model.invoke(prompt)
    return reflection

@tool("document_reflector", return_direct=True)
def reflect_document(output: str) -> str:
    """
    REFLECTION TOOL: Evaluates the success of the document writing process.
    """
    prompt = f"Evaluate the following document writing output for success and correctness:\n{output}\nProvide feedback."
    reflection = reflection_model.invoke(prompt)
    return reflection

@tool("knowledge_graph_reflector", return_direct=True)
def critique_knowledge_graph(output_json: str) -> str:
    """
    CRITIQUE TOOL: Evaluates the quality of the created knowledge graph.
    """
    prompt = (
        f"Evaluate the following knowledge graph for accuracy, completeness, and proper structuring:\n{output_json}\nProvide detailed feedback and suggestions for improvements if necessary. Critique the knowledge graph that is provided and suggest improvements and recommendations and identify if there's anything missing or incorrect. Provide detailed recommendations, including requests for reprocessing the data and re analyzing the inputs."
    )
    critique = critique_model.invoke(prompt)
    return critique

@tool("visualization_reflector", return_direct=True)
def reflect_visualization(state) -> str:
    """
    REFLECTION TOOL: Provides feedback based on the visualization check results.
    """
    check_result = state.get("check_result", "")
    prompt = (
        f"Based on the following visualization check results, identify issues and suggest improvements:\n{check_result}\n"
        "Provide detailed feedback and recommendations."
    )
    reflection = reflection_model.invoke(prompt)
    return reflection


@tool("visualization_checker", return_direct=True)
def check_visualization(image_path: str) -> str:
    """
    CHECK TOOL: Assesses the quality of the knowledge graph visualization.
    """
    prompt = (
        f"Assess the quality of the knowledge graph visualization located at {image_path}. "
        "Is the visualization clear, accurate, and effectively representing the knowledge graph? "
        "Provide a concise assessment."
    )
    check_result = check_model.invoke(prompt)
    return check_result



# Set up the state
from langgraph.graph import MessagesState, START
from langgraph.types import Command, interrupt

# Set up the tool
# We will have one real tool - a search tool
# We'll also have one "fake" tool - a "ask_human" tool
# Here we define any ACTUAL tools
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode



main_tools = [extract_symbols_t, extract_texts_t, extract_lines_t, write_document_t]


# Initialize ToolNode with main tools only
tool_node = ToolNode(main_tools)



# Set up the model
# model = ChatOpenAI(temperature=0, streaming=True)
model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0, streaming=True)

from pydantic import BaseModel


# We are going "bind" all tools to the model
# We have the ACTUAL tools from above, but we also need a mock tool to ask a human
# Since `bind_tools` takes in tools but also just tool definitions,
# We can define a tool definition for `ask_human`
class AskHuman(BaseModel):
    """Ask the human a question"""

    question: str



model = model.bind_tools(main_tools + [AskHuman])

# Define nodes and conditional edges


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return END
    # If tool call is asking Human, we return that node
    # You could also add logic here to let some system know that there's something that requires Human input
    # For example, send a slack message, etc
    elif last_message.tool_calls[0]["name"] == "AskHuman":
        return "ask_human"
    # Otherwise if there is, we continue
    else:
        return "action"

# Function to aggregate extracted data and trigger document writing
def aggregate_and_write(state):
    # Extract the latest detections
    symbols = extract_symbols_t("")
    symbols_json = json.dumps(symbols, indent=2)
    symbol_reflection = reflect_symbols(symbols_json)
    logger.info(f"Symbol Reflection: {symbol_reflection}")

    texts = extract_texts_t("")
    texts_json = json.dumps(texts, indent=2)
    text_reflection = reflect_texts(texts_json)
    logger.info(f"Text Reflection: {text_reflection}")

    lines = extract_lines_t("")
    lines_json = json.dumps(lines, indent=2)
    line_reflection = reflect_lines(lines_json)
    logger.info(f"Line Reflection: {line_reflection}")

    # Aggregate the data
    processed_data = {
        "symbols": symbols,
        "texts": texts,
        "lines": lines
    }
    # processed_data = {
    #     "symbols": [],
    #     "texts": [],
    #     "lines": []
    # }

    wrapped_processed_data = {"processed_data": processed_data}

    # # Write to document
    write_response = write_document_t(wrapped_processed_data)
    document_reflection = reflect_document(write_response)
    logger.info(f"Document Reflection: {document_reflection}")

    
    # Create the knowledge graph
    knowledge_graph = create_knowledge_graph(processed_data)
    knowledge_graph_json = json.dumps(knowledge_graph, indent=2)
    logger.info(f"Knowledge Graph Created: {knowledge_graph_json}")

    # # Reflect on the knowledge graph
    # knowledge_graph_critique = critique_knowledge_graph(knowledge_graph_json)
    # state["knowledge_graph_critique"] = knowledge_graph_critique
    # logger.info(f"Knowledge Graph Reflection: {knowledge_graph_critique}")

    # # Optionally, decide based on reflection whether to proceed or take corrective actions
    # if "issues" in knowledge_graph_critique.content.lower() or "suggestion" in knowledge_graph_critique.content.lower():
    #     # Implement logic to handle reflections, e.g., re-run extraction or notify human
    #     # For simplicity, we'll log the reflection
    #     logger.warning("Knowledge Graph has issues. Consider reviewing the reflections.")
    
    return {"messages": [HumanMessage(content=write_response)]}



def create_knowledge_graph(processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Aggregate all data from the text, line and symbol detectors"""
    # NOTE: This currently points to the json file, adjust later
    with open("/Users/hivamoh/Desktop/IntuigenceAI/2/2_graph_network.json", 'r') as file:
        data = json.load(file)
    return data

def visualize_knowledge_graph(processed_data: Dict[str, Any]) -> Dict[str, Any]:
    """Takes the data and construct a knowledge graph of the P&ID"""
    # NOTE: This currently points to the json file, adjust later
    image = "/Users/hivamoh/Desktop/IntuigenceAI/2/2_graph_network_overlay.jpg"
    return  {"image_path": image}

# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# We define a node to ask the human
def ask_human(state):
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    location = input("What is the p&id path: ")
    tool_message = [{
        "tool_call_id": tool_call_id,
        "role": "tool",  # Corrected key
        "content": location
    }]
    return {"messages": tool_message}

# Add a new node for reflecting on the knowledge graph
def critique_knowledge_graph_step(state):
    with open("/Users/hivamoh/Desktop/IntuigenceAI/2/2_aggregated_detections.json", 'r') as file:
        knowledge_graph_json = json.load(file)
    knowledge_graph_critique = critique_knowledge_graph(f"{knowledge_graph_json['detailed_results']}")
    logger.info(f"Knowledge Graph Critique: {knowledge_graph_critique}")
    return {"finished": True}
    # if "issue" in knowledge_graph_critique.content.lower() or "improve" in knowledge_graph_critique.content.lower():
    #     logger.warning("Issues detected in the Knowledge Graph. Initiating corrective actions.")
    #     # Set a flag to indicate corrective actions are needed
    #     return {"needs_correction": True}
    # else:
    #     logger.info("Knowledge Graph is satisfactory.")
    #     # Set a flag to indicate the process is finished
    #     return {"finished": True}


def check_visualization_step(state):
    """
    Hardcoded CHECK TOOL: Assesses the quality of the knowledge graph visualization.
    Always returns that the visualization is satisfactory.
    """
    logger.info("Hardcoded: Visualization is satisfactory.")
    return {"proceed_visualization": True}


#####-----------------****** GET HUMAN FEEDBACK AGENT ******-----------------#####
def get_human_feedback(state):
    """
    Prompt the user to provide feedback on the final knowledge graph.

    Args:
        state (PidState): The current state of the workflow.

    Returns:
        PidState: The updated state with user feedback.
    """
    user_input = input("Is the final knowledge graph satisfactory? (yes/no): ")
    # # Update the state with the user's feedback
    # state['user_feedback'] = user_input.lower()
    if user_input in ['yes', 'y']:
        needs_correction = False
    elif user_input in ['no', 'n']:
        needs_correction = True
    else:
        print("Invalid input. Please respond with 'yes' or 'no'.")
        # Optionally, you can loop until valid input is received
        return get_human_feedback(state)
    print(user_input)
    return {"user_feedback": user_input}

from langgraph.graph import StateGraph, START, END




# Define a node to end the visualization process if no issues
def end_visualization(state):
    logger.info("Visualization is satisfactory. Proceeding to the next steps.")
    return {"finished": True}
# Build the graph

from langgraph.graph import END, StateGraph

# Define a new graph
workflow = StateGraph(MessagesState)

# Define the three nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_node("ask_human", ask_human)
workflow.add_node("aggregate_write", aggregate_and_write)
workflow.add_node("create_knowledge_graph", create_knowledge_graph)  
workflow.add_node("visualize_knowledge_graph", visualize_knowledge_graph) 
workflow.add_node("check_visualization_step", check_visualization_step)
workflow.add_node("critique_knowledge_graph", critique_knowledge_graph_step)
# workflow.add_node("check_visualization", check_visualization)
# workflow.add_node("reflect_visualization", reflect_visualization)
# workflow.add_node("end_visualization", end_visualization)
workflow.add_node("human_feedback", get_human_feedback) 



# Add edges to integrate the new nodes into the workflow
# After 'visualize_knowledge_graph', go to 'check_visualization_step'
workflow.add_edge("visualize_knowledge_graph", "check_visualization_step")
workflow.add_edge("create_knowledge_graph", "critique_knowledge_graph")

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # {
    #     # "END": END,           # Terminate the workflow
    #     "ask_human": "ask_human",  # Move to 'ask_human' node
    #     "action": "action"    # Move to 'action' node
    # }
)
# After we get back the human response, we go back to the agent
workflow.add_edge("ask_human", "agent")

# Edge from `action` to `aggregate_write`
workflow.add_edge("action", "aggregate_write")

# Edge from `aggregate_write` back to `agent`
workflow.add_edge("aggregate_write", "agent")

# Add conditional edges after 'critique_knowledge_graph' node
workflow.add_conditional_edges(
    "critique_knowledge_graph",
    lambda state: "visualize_knowledge_graph" if state.get("needs_correction") else  "visualize_knowledge_graph",
    {"visualize_knowledge_graph": "visualize_knowledge_graph", "visualize_knowledge_graph": "visualize_knowledge_graph"}
    # lambda state: "action" if state.get("needs_correction") else  "visualize_knowledge_graph",
    # {"action": "action", "visualize_knowledge_graph": "visualize_knowledge_graph"}
)

# # Define conditional edges based on user_feedback
# workflow.add_conditional_edges(
#     "human_feedback",
#     lambda state: "END" if state.get("user_feedback") in ['yes', 'y'] else "END",
#     {
#         "END": END,
#         "END": END
#     }
# )
workflow.add_edge("human_feedback", END)

# workflow.add_edge("end_visualization", END)

# Edge from 'reflect_visualization' back to 'visualize_knowledge_graph' or other corrective actions
# workflow.add_edge("reflect_visualization", "visualize_knowledge_graph")

# Conditional edges based on the hardcoded check result
workflow.add_conditional_edges(
    "check_visualization_step",
    # lambda state: "end_visualization" if state.get("proceed_visualization") else "reflect_visualization",
    # {
    #     "end_visualization": "human_feedback",
    #     "reflect_visualization": "reflect_visualization"  
    # }
    lambda state: "end_visualization" if state.get("proceed_visualization") else "human_feedback",
    {
        "end_visualization": "human_feedback",
        "human_feedback": "human_feedback"  
    }
)

workflow.add_edge("critique_knowledge_graph", "visualize_knowledge_graph")

workflow.add_edge("aggregate_write", "create_knowledge_graph")



# Set up memory
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
# We add a breakpoint BEFORE the `ask_human` node so it never executes
app = workflow.compile(checkpointer=memory)
# from IPython.display import Image, display
# display(Image(app.get_graph().draw_mermaid_png()))


config = {"configurable": {"thread_id": "2"}}
for event in app.stream( 
    {
        "messages": [
            (
                "user",
                # "Use the tools to ask the user for the p&id path and then use the path to call the tools to extract the symbols, texts and lines from the given p&ID. Then aggregate the data given from the process tools. Then create a knowledge graph and after criticizing it, visualize it and check it. Make sure to ask for user feedaback and if they said that they are satisfied by the final knowledge graph, end the loop. Make sure you are using the right tools for each stage and reflect after each step.",
                "Use the tools to ask the user for the p&id path and then use the path to call the tools to extract the symbols, texts and lines from the given p&ID.",
            )
        ]
    },
    config,
    stream_mode="values",
):
    
    if "__end__" not in event:
        print(event)
        print("----")
    print(event["messages"][-1])