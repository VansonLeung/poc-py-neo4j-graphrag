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

# Initialize shared components
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

drive_embedder = OpenAIEmbeddings(api_key="abc", model="moka-ai/m3e-base", base_url="http://localhost:18000/v1")

stories = [
    {
        "id": "jochi-orda",
        "article": """
Orda, eldest son of Jochi, inherited the windswept steppes north of the Irtysh River. Guided by the memory of his father, Orda forged alliances with forest tribes and commissioned caravans to shuttle grain toward the Volga. His sister Saran advised him through coded songs relayed by riders, ensuring that Orda's warriors never faced a battle unprepared.
""",
    },
    {
        "id": "jochi-batu",
        "article": """
Batu, second son of Jochi, established Sarai beside the lower Volga. He relied on his cousin Berke to negotiate with Persian merchants while Batu's wife Yelun curated an archive of conquered laws. Together they orchestrated winter encampments so that cavalry reserves could strike westward at spring thaw.
""",
    },
    {
        "id": "jochi-shiban",
        "article": """
Shiban, the strategist son of Jochi, preferred river ambushes. He mapped tributaries around the Syr Darya and taught his daughter Altani to command scout boats. When drought threatened, Shiban traded salt for timber with Khwarezm artisans, weaving diplomacy into logistics.
""",
    },
    {
        "id": "jochi-tangqut",
        "article": """
Tangqut, contemplative son of Jochi, adopted Buddhist advisers from Dunhuang. He raised stupas along caravan roads and tasked his son Mergen with protecting monks who documented steppe treaties. Tangqut believed spiritual patrons kept the clans focused on shared purpose.
""",
    },
    {
        "id": "jochi-berke",
        "article": """
Berke, devout son of Jochi, embraced Islam after trading in Bukhara. He invited jurist Safiya to teach sharia to Tatar chiefs and installed cisterns across the Caucasus. Berke's nephew Arslan carried letters of safe passage, binding Black Sea traders to the Golden Horde.
""",
    },
    {
        "id": "jochi-berkecousin",
        "article": """
Kuli, an adopted nephew in Jochi's line, specialized in siege workshops. He paired with engineer Sorghagtani to refit captured Chinese trebuchets. Kuli's spouse Bayan sketched each device so frontier forges could reproduce them in haste.
""",
    },
    {
        "id": "jochi-sasqa",
        "article": """
Sasqa, youngest son of Jochi, served as envoy between the Rus princes and steppe councils. He traveled with translator Nargui and brewed birch sap elixirs to gift wary dukes. Sasqa's charisma gave his brother Batu the time needed to reorganize cavalry wings.
""",
    },
    {
        "id": "jochi-uykit",
        "article": """
Uykit, a healer descended from Jochi, ran mobile infirmaries. She trained apprentices to stitch wounds during thunder snowstorms and carried herbal compendiums compiled by Lady Jessica's distant correspondents. Uykit proved that compassion could fortify a march as surely as armor.
""",
    },
    {
        "id": "jochi-darma",
        "article": """
Darma, scholarly grandson of Jochi, catalogued every clan banner in lacquered tablets. His son Tumen memorized the colors to signal feints in night battles. Darma corresponded with Venetian artisans, trading lapis for glass so scouts could view star charts through crystal disks.
""",
    },
    {
        "id": "jochi-qarqal",
        "article": """
Qarqal, storm-chasing descendant of Jochi, rode ahead of thunderheads to chart safe crossings. His partner Hulan interpreted cloud patterns, and together they warned caravan masters of flash floods. Qarqal's niece Yiska later taught these techniques to young scouts across the Ulus of Jochi.
""",
    },
]

prompt_template = """
Given an article:

```
{article}
```

Analyze the following article and extract:

- patterns: A complete list of lists representing entity relationships (e.g., [["Lady Jessica", "MENTORED", "Genghis Khan"]])

Return the result as a JSON object with keys: patterns.
"""

failed_stories = []

for story in stories:
    article = story["article"].strip()
    story_id = story["id"]

    prompt = prompt_template.format(article=article)
    response = llm.invoke(prompt)

    node_types = set()
    relationship_types = set()

    try:
        # Invoke the LLM to get schema
        result = json.loads(response.content)

        # Extract the generated schema
        patterns = result.get("patterns", [])
        patterns = [tuple(p) for p in patterns]  # Convert to list of tuples

        # Extract node types and relationship types from patterns
        for p in patterns:
            node_types.add(p[0])
            node_types.add(p[2])
            relationship_types.add(p[1])

        # Ensure all entities in patterns are in node_types
        for p in patterns:
            if p[0] not in node_types:
                node_types.append(p[0])
            if p[2] not in node_types:
                node_types.append(p[2])

    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"Skipping {story_id}: unable to parse LLM response ({exc}). Raw response: {response.content}")
        failed_stories.append(story_id)
        continue

    try:
        print(f"Generated schema for {story_id}:")
        print("node_types =", node_types)
        print("relationship_types =", relationship_types)
        print("patterns =", patterns)

        kg_builder = SimpleKGPipeline(
            llm=llm,
            driver=driver,
            embedder=drive_embedder,
            schema={
                "node_types": node_types,
                "relationship_types": relationship_types,
                "patterns": patterns,
            },
            on_error="IGNORE",
            from_pdf=False,
        )

        asyncio.run(kg_builder.run_async(text=article))

        vector = drive_embedder.embed_query(article)

        upsert_vectors(
            driver,
            ids=[story_id],
            embedding_property="vectorProperty",
            embeddings=[vector],
            entity_type=EntityType.NODE,
        )
    except Exception as exc:
        print(f"Processing failed for {story_id}: {exc}")
        failed_stories.append(story_id)
        continue

if failed_stories:
    print("The following stories failed:", failed_stories)
else:
    print("All Jochi offspring stories have been ingested and indexed.")

driver.close()
