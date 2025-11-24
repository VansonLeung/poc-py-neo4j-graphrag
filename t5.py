from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.indexes import upsert_vectors
from neo4j_graphrag.types import EntityType

NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"

# Connect to the Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Create an Embedder object
embedder = OpenAIEmbeddings(api_key="abc", model="moka-ai/m3e-base", base_url="http://localhost:18000/v1")

# Generate an embedding for some text
text = (
    "The son of Duke Leto Atreides and the Lady Jessica, Paul is the heir of House "
    "Atreides, an aristocratic family that rules the planet Caladan."
)
vector = embedder.embed_query(text)

# Upsert the vector
upsert_vectors(
    driver,
    ids=["1234"],
    embedding_property="vectorProperty",
    embeddings=[vector],
    entity_type=EntityType.NODE,
)
driver.close()