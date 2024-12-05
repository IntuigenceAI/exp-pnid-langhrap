import networkx as nx
from typing import List, Dict

def build_knowledge_graph(classified_components: List[Dict], relationships: List[Dict]) -> Dict:
    """
    Constructs a knowledge graph from classified components and relationships.

    Parameters:
        classified_components (List[Dict]): List of classified components.
        relationships (List[Dict]): List of relationships between components.

    Returns:
        Dict: A dictionary representation of the knowledge graph.
    """
    try:
        # Initialize a directed graph
        graph = nx.DiGraph()

        # Add nodes to the graph
        for component in classified_components:
            graph.add_node(
                component["id"],
                label=component["label"],
                type=component["type"],
                properties=component.get("properties", {})
            )

        # Add edges (relationships) to the graph
        for relationship in relationships:
            source = relationship["source"]
            target = relationship["target"]
            relationship_type = relationship["type"]
            graph.add_edge(source, target, type=relationship_type)

        # Convert the graph to a dictionary for JSON serialization
        graph_dict = nx.node_link_data(graph)
        return graph_dict
    except Exception as e:
        raise RuntimeError(f"Error building knowledge graph: {e}")


if __name__ == "__main__":
    # Example usage
    classified_components = [
        {"id": "C1", "label": "Valve", "type": "Control", "properties": {"pressure": "100psi"}},
        {"id": "C2", "label": "Pump", "type": "Centrifugal", "properties": {"flow_rate": "50gpm"}},
    ]
    relationships = [
        {"source": "C1", "target": "C2", "type": "Controlled by"},
    ]

    knowledge_graph = build_knowledge_graph(classified_components, relationships)
    print(knowledge_graph)
