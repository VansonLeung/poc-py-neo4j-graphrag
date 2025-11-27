import asyncio
import json
import neo4j
import os
from openai import OpenAI
from t22 import create_entity_node, create_relationship, create_document, create_document_chunk

# ANSI color codes for logging
RESET = "\033[0m"
BLUE = "\033[34m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"

# Neo4j connection details (adjust as needed)
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")  # Replace with your actual credentials

# Connect to the Neo4j database
driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)

# Initialize the OpenAI client
openai_client = OpenAI(
    api_key="",
    base_url="http://localhost:18000/v1",
)

def llm_create(prompt):
  return openai_client.chat.completions.create(
      model="Qwen3-4B-Instruct-2507-4bit",
      messages=[
          {"role": "user", "content": prompt}
      ],
      max_tokens=1000,
      temperature=0.0,
  )

def embedding_create(text):
  return openai_client.embeddings.create(
      input=text,
      model="moka-ai/m3e-base",
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
response = llm_create(prompt)
result = json.loads(response.choices[0].message.content)

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

print(f"{BLUE}Generated schema for Mongolian story:{RESET}")
print(f"{GREEN}node_types = {node_types}{RESET}")
print(f"{RED}relationship_types = {relationship_types}{RESET}")
print(f"{YELLOW}patterns = {patterns}{RESET}")

# Create an Embedder function using OpenAI SDK
def get_embedding(text):
    response = embedding_create(text)
    return response.data[0].embedding

# Get embedding for the article
print(f"{BLUE}Generating embedding for the article...{RESET}")
embedding = get_embedding(article)
print(f"{GREEN}Embedding generated successfully.{RESET}")

# Create document and chunk in Neo4j
print(f"{BLUE}Creating document in Neo4j...{RESET}")
document_id = create_document(driver, "article", "school_roles.txt")
print(f"{GREEN}Document created with ID: {document_id}{RESET}")

print(f"{BLUE}Creating chunk with embedding...{RESET}")
chunk_id = create_document_chunk(driver, document_id, 0, article, embedding)
print(f"{GREEN}Chunk created with ID: {chunk_id}{RESET}")

# Create entities and relationships based on the dynamic schema
print(f"{BLUE}Creating entities and relationships from extracted patterns...{RESET}")
created_entities = set()  # Track created entities to avoid duplicates
for pattern in patterns:
    entity1, rel, entity2 = pattern
    
    # Create entity nodes if not already created
    if entity1 not in created_entities:
        create_entity_node(driver, chunk_id, entity1)
        created_entities.add(entity1)
        print(f"{YELLOW}Created entity: {entity1}{RESET}")
    
    if entity2 not in created_entities:
        create_entity_node(driver, chunk_id, entity2)
        created_entities.add(entity2)
        print(f"{YELLOW}Created entity: {entity2}{RESET}")
    
    # Create relationship
    create_relationship(driver, entity1, rel, entity2)
    print(f"{YELLOW}Created relationship: {entity1} -[:{rel}]-> {entity2}{RESET}")

print(f"{GREEN}All entities and relationships created successfully.{RESET}")








print(f"{GREEN}Knowledge graph for Mongolian story built successfully with dynamic schema and vector upserted.{RESET}")

