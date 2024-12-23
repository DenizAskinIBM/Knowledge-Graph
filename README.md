# Knowledge-Graph
Creating and Querying Neo4J Knowledge Graphs with LLMs

To run, execute the following commands on your terminal to initiate Neo4J:

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
