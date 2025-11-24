import json
from neo4j_graphrag.llm import OpenAILLM

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

# Prompt for the LLM
prompt = f"""
Analyze the following article and extract:

- node_types: A list of unique entity types (e.g., ["Person", "House", "Planet"])
- relationship_types: A list of unique relationship types (e.g., ["PARENT_OF", "HEIR_OF", "RULES"])
- patterns: A list of tuples representing common patterns (e.g., [("Person", "PARENT_OF", "Person"), ("Person", "HEIR_OF", "House")])

Return the result as a JSON object with keys: node_types, relationship_types, patterns.

Article:
{article}
"""

# Invoke the LLM
response = llm.invoke(prompt)
result = json.loads(response.content)

# Extract the generated schema
node_types = result.get("node_types", [])
relationship_types = result.get("relationship_types", [])
patterns = result.get("patterns", [])
patterns = [tuple(p) for p in patterns]  # Convert to list of tuples

# Print the results
print("node_types =", node_types)
print("relationship_types =", relationship_types)
print("patterns =", patterns)
