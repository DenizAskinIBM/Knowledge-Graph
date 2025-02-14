import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE"))
import requests
from bs4 import BeautifulSoup
import networkx as nx
from networkx.algorithms import community
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set your OpenAI API key (ensure it's set in your environment or replace the fallback)

def call_openai_api(prompt: str, model: str = "gpt-4", max_tokens: int = 300) -> str:
    """
    Calls the OpenAI API using the Chat Completion endpoint with GPT-4.
    
    :param prompt: The prompt to send.
    :param model: The model name (default: "gpt-4").
    :param max_tokens: Maximum tokens in the response.
    :return: The generated text from GPT-4.
    """
    try:
        response = client.chat.completions.create(model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error during API call: {e}"

########################################################################
# GraphRAG-Style Indexer Using OpenAI GPT-4 with Parallel API Calls
########################################################################
class GraphRAGIndexer:
    def __init__(self):
        self.graph = nx.Graph()
        self.community_summaries = {}  # Mapping from community ID to summary
        self.node_to_community = {}    # Mapping from each node to its community ID

    def process_documents(self, text_corpus: str, chunk_size: int = 500) -> None:
        # 1. Split the text into chunks.
        text_units = self._chunk_text(text_corpus, chunk_size)
        # 2. Extract entities and relationships from each chunk in parallel.
        extracted_data = self._extract_entities_and_relations(text_units)
        # 3. Build the knowledge graph.
        self._build_graph(extracted_data)
        # 4. Detect communities.
        self._detect_communities()
        # 5. Generate community summaries in parallel.
        self._generate_community_summaries()

    def _chunk_text(self, text: str, chunk_size: int) -> list:
        """Splits text into fixed-size chunks."""
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def _extract_entities_and_relations(self, text_units: list) -> list:
        """
        For each text chunk, call GPT-4 to extract entities and relationships.
        Expected output format (as a string):
        
            {
              "entities": ["Entity1", "Entity2", ...],
              "relations": [("Entity1", "relationship", "Entity2"), ...]
            }
        """
        results = []
        # Double curly braces to produce literal braces in the formatted string.
        prompt_template = (
            "Extract entities and relationships from the following text in the format:\n"
            "{{'entities': [<list_of_entities>], 'relations': [(entity1, relation, entity2), ...]}}\n"
            "Text: {}"
        )
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_unit = {
                executor.submit(call_openai_api, prompt_template.format(unit), "gpt-4", 300): unit
                for unit in text_units
            }
            for future in as_completed(future_to_unit):
                try:
                    response = future.result()
                    # Warning: using eval() is insecure. Consider using json.loads with a JSON response.
                    try:
                        data = eval(response)
                    except Exception:
                        data = {"entities": [], "relations": []}
                except Exception:
                    data = {"entities": [], "relations": []}
                results.append(data)
        return results

    def _build_graph(self, extracted_data: list) -> None:
        """Builds a graph from the extracted entities and relationships."""
        for er in extracted_data:
            for entity in er.get("entities", []):
                self.graph.add_node(entity)
            for rel in er.get("relations", []):
                if len(rel) == 3:
                    self.graph.add_edge(rel[0], rel[2], label=rel[1])

    def _detect_communities(self) -> None:
        """
        Detects communities using NetworkX's greedy modularity community detection
        and assigns each node a community ID.
        """
        if self.graph.number_of_nodes() == 0:
            return
        comms = list(community.greedy_modularity_communities(self.graph))
        for idx, comm in enumerate(comms):
            for node in comm:
                self.node_to_community[node] = idx

    def _generate_community_summaries(self) -> None:
        """
        Generates a summary for each community by passing the list of nodes to GPT-4.
        This process is parallelized using a ThreadPoolExecutor.
        """
        # Organize nodes by community ID.
        communities_dict = {}
        for node, comm_id in self.node_to_community.items():
            communities_dict.setdefault(comm_id, []).append(node)
        summaries = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_comm = {
                executor.submit(
                    call_openai_api,
                    f"Please summarize the following community of related entities: {nodes}",
                    "gpt-4",
                    300
                ): comm_id
                for comm_id, nodes in communities_dict.items()
            }
            for future in as_completed(future_to_comm):
                comm_id = future_to_comm[future]
                try:
                    summaries[comm_id] = future.result()
                except Exception:
                    summaries[comm_id] = "No summary available."
        self.community_summaries = summaries

########################################################################
# Query Engine
########################################################################
class QueryEngine:
    def __init__(self, indexer: GraphRAGIndexer):
        self.indexer = indexer

    def global_search(self, query: str) -> str:
        """
        Uses community summaries to answer a high-level query.
        """
        # Truncate each summary to reduce prompt length.
        summarized_communities = {cid: summary[:500] for cid, summary in self.indexer.community_summaries.items()}
        prompt = (
            f"Using the following community summaries (truncated for brevity): {summarized_communities}\n"
            f"Answer the following question: {query}"
        )
        return call_openai_api(prompt, model="gpt-4", max_tokens=300)

    def local_search(self, query: str) -> str:
        """
        Uses the raw knowledge graph to answer a query.
        """
        # Limit the graph representation to a subset of nodes to reduce prompt size.
        sample_nodes = list(self.indexer.graph.nodes())[:50]
        graph_repr = {node: list(self.indexer.graph.neighbors(node)) for node in sample_nodes}
        prompt = (
            f"Using the following knowledge graph (subset): {graph_repr}\n"
            f"Answer the following question: {query}"
        )
        return call_openai_api(prompt, model="gpt-4", max_tokens=300)

########################################################################
# Main Method (Example Usage)
########################################################################
def main():
    # Scrape text from a Wikipedia page (example: Elon Musk)
    url = "https://en.wikipedia.org/wiki/Elon_Musk"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text()

    # Build the GraphRAG index.
    indexer = GraphRAGIndexer()
    indexer.process_documents(text)

    # Create a query engine.
    query_engine = QueryEngine(indexer)

    # Example: Local search.
    local_query = "What companies were founded by Elon Musk?"
    local_answer = query_engine.local_search(local_query)
    print("Local Search Answer:")
    print(local_answer)

    # Example: Global search.
    global_query = "Summarize Elon Musk's major accomplishments."
    global_answer = query_engine.global_search(global_query)
    print("\nGlobal Search Answer:")
    print(global_answer)

if __name__ == "__main__":
    main()
