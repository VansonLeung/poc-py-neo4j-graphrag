from neo4j import GraphDatabase
from graphrag_m import GraphRAG
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever

NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"
INDEX_NAME = "vector-index-name"

# Connect to the Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Create an Embedder object
embedder = OpenAIEmbeddings(api_key="abc", model="moka-ai/m3e-base", base_url="http://localhost:18000/v1")

# Initialize the retriever
retriever = VectorRetriever(driver, INDEX_NAME, embedder)

# Instantiate the LLM
llm = OpenAILLM(api_key="abc", model_name="Qwen3-4B-Instruct-2507-4bit", base_url="http://localhost:18000/v1", model_params={"temperature": 0})

# Instantiate the RAG pipeline
rag = GraphRAG(retriever=retriever, llm=llm)

# List of queries about the Mongolians
queries = [
    "Who is Genghis Khan?",
    "What did Genghis Khan accomplish?",
    "Who are the sons of Genghis Khan?",
    "What is the Mongol Empire?",
    "Who mentors Genghis Khan?",  # To test cross-relationship
    "How did Genghis Khan unite the tribes?",
]

# Perform searches for each query
for query_text in queries:
    print(f"\nQuery: {query_text}")
    response = rag.search(query_text=query_text, retriever_config={"top_k": 5}, return_context=True)
    print(f"Answer: {response.answer}")
    if hasattr(response, 'retriever_result') and response.retriever_result:
        print(f"Context items: {len(response.retriever_result.items)}")

driver.close()