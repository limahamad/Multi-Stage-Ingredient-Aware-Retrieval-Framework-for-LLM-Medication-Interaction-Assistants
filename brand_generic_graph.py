import json
from pathlib import Path
from typing import Dict, List, Optional


class BrandGenericGraph:
    def __init__(self, graph_path: str):
        self.graph_path = Path(graph_path)

        with open(self.graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.nodes = {node["id"]: node for node in data["nodes"]}
        self.name_to_ids: Dict[str, List[str]] = {}
        self.adj: Dict[str, List[Dict]] = {}

        for node in data["nodes"]:
            key = node["name"].strip().lower()
            self.name_to_ids.setdefault(key, []).append(node["id"])

        for edge in data["edges"]:
            self.adj.setdefault(edge["source"], []).append(edge)

    def find_node_ids_by_name(self, name: str) -> List[str]:
        return self.name_to_ids.get(name.strip().lower(), [])

    def get_node(self, node_id: str) -> Dict:
        return self.nodes[node_id]

    def normalize_brand_to_generic(self, name: str) -> Optional[str]:
        """
        Return the canonical generic name if a brand is found.
        If the input is already generic, return it as is.
        If not found, return None.
        """
        node_ids = self.find_node_ids_by_name(name)

        for node_id in node_ids:
            node = self.get_node(node_id)

            # if already generic
            if node["type"] == "generic":
                return node["name"]

            # if brand, follow maps_to edge
            if node["type"] == "brand":
                for edge in self.adj.get(node_id, []):
                    if edge["relation"] == "maps_to":
                        target_node = self.get_node(edge["target"])
                        if target_node["type"] == "generic":
                            return target_node["name"]

        return None