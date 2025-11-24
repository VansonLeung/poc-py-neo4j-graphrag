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

# Sample long article (replace with actual article)
article = """
The son of Duke Leto Atreides and the Lady Jessica, Paul is the heir of House Atreides, an aristocratic family that rules the planet Caladan. The Atreides family has been granted stewardship of the desert planet Arrakis, also known as Dune, by the Emperor. Arrakis is the only source of the spice melange, a substance that extends life and enhances mental abilities. Paul's mother, Lady Jessica, is a member of the Bene Gesserit, a powerful sisterhood with advanced training in mental and physical disciplines. On Arrakis, Paul encounters the Fremen, the native people of the desert, who have adapted to the harsh environment. The Fremen leader, Stilgar, becomes an ally to Paul. Paul also learns about the prophecy of the Muad'Dib, a messianic figure who will lead the Fremen to freedom. The Harkonnens, rivals of the Atreides, plot to overthrow them and seize control of Arrakis. Baron Vladimir Harkonnen, the head of the family, orchestrates the betrayal. Paul's father, Duke Leto, is killed in the coup, and Paul and his mother flee into the desert. There, Paul undergoes a transformation, embracing the Fremen culture and becoming their leader. He marries Chani, a Fremen woman, and fathers a child. The story culminates in Paul's rise as the Muad'Dib, leading a revolution against the Empire and the Harkonnens.
"""

# Prompt for the LLM to generate schema
prompt = f"""
Analyze the following article and extract:

- node_types: A list of unique entity types (e.g., ["Person", "House", "Planet"])
- relationship_types: A list of unique relationship types (e.g., ["PARENT_OF", "HEIR_OF", "RULES"])
- patterns: A list of tuples representing common patterns (e.g., [("Person", "PARENT_OF", "Person"), ("Person", "HEIR_OF", "House")])

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

print("Generated schema:")
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

# Generate an embedding for some text and upsert
vector = embedder.embed_query(article)

# Upsert the vector
upsert_vectors(
    driver,
    ids=["1234"],
    embedding_property="vectorProperty",
    embeddings=[vector],
    entity_type=EntityType.NODE,
)

driver.close()

print("Knowledge graph built successfully with dynamic schema and vector upserted.")