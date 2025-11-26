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

    graph_strategy = {
        "top_k": 5,
        "graph_strategy": {"max_iterations": 4, "thought_prefix": "Hop"},
    }

    queries = [
        # "Which alliances did Orda depend on?",
        # "How did Batu coordinate campaigns?",
        # "Who handled healing efforts for the Jochid forces?",
        "Ultimately whose knowledge might be transferred to Jochi through his father (not Jochi's descendents)?",
        "Ultimately whose knowledge might be transferred to Jochi through his father (not Jochi's descendents)? Find out and explain the story of his/her.",
    ]

    for query in queries:
        print(f"\n[Graph Search] {query}")
        result = rag.search_nodes_relationships_only(
            query_text=query,
            retriever_config=graph_strategy,
            return_context=True,
            response_fallback="No graph evidence found for this question.",
        )
        print(f"Answer:\n{result.answer}")
        if result.retriever_result:
            print(f"Context items: {len(result.retriever_result.items)}")

    driver.close()


if __name__ == "__main__":
    main()
