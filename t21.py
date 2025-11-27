import neo4j

# Neo4j connection details (adjust as needed)
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")  # Replace with your actual credentials

def main():
    driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)
    
    # Define the embedding list (truncated in the query, but using provided values)
    embedding = [
        0.011327064596116543, 0.023420892655849457, 0.03523365035653114,
        0.005819568410515785, 0.03588000312447548, -0.030048348009586334,
        0.00917533878237009  # Add more values if available, or use as is
    ]
    
    try:
        with driver.session() as session:
            # Create both nodes and the relationship in a single transaction
            session.run(
                """
                CREATE (d:Document {document_type: $doc_type, path: $path})
                CREATE (c:Chunk {index: $index, text: $text, embedding: $embedding})
                CREATE (c)-[:FROM_DOCUMENT]->(d)
                """,
                doc_type="inline_text", path="document.txt",
                index=0, text="Loren Ipsum Hello world (+ 50 words here)", embedding=embedding
            )
            print("Created Document node, Chunk node, and FROM_DOCUMENT relationship.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.close()
        print("Driver closed.")

if __name__ == "__main__":
    main()