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
2.5校內發展推動者的職能
一般人都認為校長、小學課程統籌主任、資深教師、科主任、教師和
學校圖書館主任是協助學生學習的發展推動者。
2.5.1校長
有關校長的職能，我們的建議是：
1.制定課程、教學和評估的策略
2.釐定發展的目標和優先次序
3.營造良好的學習環境
4.訂定教師專業發展計劃，以培育教職員在課程和教學方面的領導能力
5.制定推行學校組織改革的總目標和階段目標
6.注重學與教的質素而不是數量
7.為教師創造空間及給予教師足夠時間以發展課程
8.妥善管理資源，增加資源運用的透明度
9.協調各部門和支持各部門的自主
10.當校內改革取得進展和改善時，給予表揚
11.與教職員有效地溝通
12.撤除障礙
13.分享知識和經驗
14.與家長充分溝通
2.5.2小學課程統籌主任
有關小學課程統籌主任的職能，我們的建議是：
1.協助校長領導學校的整體課程規劃、推行及評鑑有關計劃；
2.輔助校長規劃並統籌評估政策和推行評估工作；
3.統籌各學習領域／科組，協力推動課程更新重點1；
4.領導教師或專責人員改善學與教策略；
5.推廣專業交流文化；及
6.負責適量的教學工作（約相等於校內教師平均教擔的50%），以試行各
項策略，從而進一步促進課程發展。
"""

# Prompt for the LLM to generate schema
prompt = f"""
Given an article:

```
{article}
```

Analyze the following article and extract:

- patterns: A complete list of lists representing entity relationships (e.g., [["Lady Jessica", "MENTORED", "Genghis Khan"]])

Return the result as a JSON object with keys: patterns.
"""



# Invoke the LLM to get schema
response = llm.invoke(prompt)
result = json.loads(response.content)

# Extract the generated schema
patterns = result.get("patterns", [])
patterns = [tuple(p) for p in patterns]  # Convert to list of tuples

# Extract node types and relationship types from patterns
node_types = set()
relationship_types = set()
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

# Generate an embedding for the article text and upsert
vector = embedder.embed_query(article)

# Upsert the vector
upsert_vectors(
    driver,
    ids=["5678"],
    embedding_property="vectorProperty",
    embeddings=[vector],
    entity_type=EntityType.NODE,
)

driver.close()

print("Knowledge graph for Mongolian story built successfully with dynamic schema and vector upserted.")

