import asyncio
import json

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.indexes import upsert_vectors
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.types import EntityType

NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"

# Connect to the Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Initialize the LLM
llm = OpenAILLM(
    api_key="abc",
    model_name="Qwen3-4B-Instruct-2507-4bit",
    base_url="http://localhost:18000/v1",
    model_params={
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
        "temperature": 0,
    },
)

# Sample long article about Mongolians (Genghis Khan)
article = """
Genghis Khan, originally named Temujin, was born in 1162 in the Mongol steppes. He united the warring Mongol tribes through alliances and conquests, becoming the founder of the Mongol Empire. His mother, Hoelun, raised him after his father's death. Genghis Khan was a brilliant military strategist, using tactics like feigned retreats and psychological warfare. He conquered vast territories, from the Sea of Japan to the Adriatic Sea. His sons, Jochi, Chagatai, Ogedei, and Tolui, played key roles in expanding the empire. Ogedei succeeded him as Khan. The Mongols under Genghis Khan promoted religious tolerance and meritocracy. His legacy includes the largest contiguous empire in history. Genghis Khan died in 1227, but his descendants continued his conquests.
"""

# Prompt for the LLM to generate schema
prompt = f"""
Analyze the following article and extract:

- node_types: A list of unique entity types (e.g., ["Person", "Empire", "Region"])
- relationship_types: A list of unique relationship types (e.g., ["FOUNDED", "SUCCEEDED", "CONQUERED"])
- patterns: A list of lists representing common patterns (e.g., [["Person", "FOUNDED", "Empire"], ["Person", "SUCCEEDED", "Person"]])

Return the result as a JSON object with keys: node_types, relationship_types, patterns.

Article:
{article}
"""

# Invoke the LLM to get schema
response = llm.invoke(prompt)
result = json.loads(response.content)

# Extract the generated schema
node_types = result.get("node_types", [])
relationship_types = result.get("relationship_types", [])
patterns = result.get("patterns", [])
patterns = [tuple(p) for p in patterns]  # Convert to list of tuples

# Ensure all entities in patterns are in node_types
for p in patterns:
    if p[0] not in node_types:
        node_types.append(p[0])
    if p[2] not in node_types:
        node_types.append(p[2])

print("Generated schema for Mongolian story:")
print("node_types =", node_types)
print("relationship_types =", relationship_types)
print("patterns =", patterns)

# Create an Embedder object
embedder = OpenAIEmbeddings(api_key="abc", model="moka-ai/m3e-base", base_url="http://localhost:18000/v1")

# Instantiate the SimpleKGPipeline with dynamic schema
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=driver,
    embedder=embedder,
    schema={
        "node_types": node_types,
        "relationship_types": relationship_types,
        "patterns": patterns,
    },
    on_error="IGNORE",
    from_pdf=False,
)

# Run the pipeline on the article text
asyncio.run(kg_builder.run_async(text=article))

# Add a single relationship between protagonists: Paul Atreides (from Dune) and Genghis Khan
def add_cross_relationship(tx):
    # Assuming nodes exist with names; adjust if properties differ
    tx.run("""
    MATCH (p:Person {name: "Paul Atreides"}), (g:Person {name: "Genghis Khan"})
    CREATE (p)-[:ALLY_OF]->(g)
    """)

with driver.session() as session:
    session.execute_write(add_cross_relationship)

# Generate an embedding for some text and upsert
text = (
    "Genghis Khan united the Mongol tribes and founded the Mongol Empire."
)
vector = embedder.embed_query(text)

# Upsert the vector
upsert_vectors(
    driver,
    ids=["5678"],
    embedding_property="vectorProperty",
    embeddings=[vector],
    entity_type=EntityType.NODE,
)

driver.close()

print("Knowledge graph for Mongolian story built successfully with dynamic schema, cross-relationship added, and vector upserted.")