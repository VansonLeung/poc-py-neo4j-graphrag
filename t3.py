from neo4j import GraphDatabase

NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"

# Connect to the Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def remove_all_nodes(tx):
    tx.run("MATCH (n) DETACH DELETE n")

def drop_vector_index(tx):
    tx.run("DROP INDEX `vector-index-name` IF EXISTS")

with driver.session() as session:
    session.execute_write(remove_all_nodes)
    session.execute_write(drop_vector_index)
    print("All nodes have been removed and the vector index has been dropped from the Neo4j database.")

driver.close()
