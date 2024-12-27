# Knowledge-Graph
Creating and Querying Neo4J Knowledge Graphs with LLMs

Compatible with Python3.11.9

To run, execute the following commands on your terminal to initiate Neo4J:

```bash
pip install podman

podman machine init

podman machine start

podman run \
    --restart always \
    --publish=7474:7474 --publish=7687:7687 \
    --env NEO4J_AUTH=neo4j/password4j \
    --env='NEO4JLABS_PLUGINS=["apoc"]' \
    --volume=$PATH:/data \
    neo4j:5.22.0
```
To generate graphs from transcripts and run queries on the whole graphs use main.py

To use GraphRAG, use graphrag.py. You can generate your own indexes for GraphRAG using graph_vector_db_generation.py
