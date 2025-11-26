from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever

from graphrag_m import GraphRAG

NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"
INDEX_NAME = "vector-index-name"


def main() -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    embedder = OpenAIEmbeddings(
        api_key="abc",
        model="moka-ai/m3e-base",
        base_url="http://localhost:18000/v1",
    )
    retriever = VectorRetriever(driver, INDEX_NAME, embedder)

    llm = OpenAILLM(
        api_key="abc",
        model_name="Qwen3-4B-Instruct-2507-4bit",
        base_url="http://localhost:18000/v1",
        model_params={"temperature": 0},
    )

    rag = GraphRAG(retriever=retriever, llm=llm)

    recursive_config = {
        "top_k": 5,
        "recursive_strategy": {
            "top_k": 4,
            "max_depth": 3,
            "max_followups": 2,
        },
    }

    queries = [
        # "How did relationships around Jochi influence logistics support?",
        # "Which relatives helped Genghis Khan maintain communication chains?",
        # "Ultimately whose knowledge might be transferred to Jochi through his father (not Jochi's descendents)?",
        # "Who is Jochi's father?",
        # "Who is Jochi's father's mentor?",
        "Who is Jochi's father's mentor and What kind of knowledge?",
    ]

    for query in queries:
        print(f"\n[Recursive Graph Search] {query}")
        result = rag.search_recursive_graph(
            query_text=query,
            retriever_config=recursive_config,
            return_context=False,
            response_fallback="No graph evidence could answer this question.",
        )
        print(f"Answer:\n{result.answer}")

    driver.close()


if __name__ == "__main__":
    main()
