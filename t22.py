import neo4j

# Neo4j connection details (adjust as needed)
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")  # Replace with your actual credentials

def create_entity_node(driver, chunk_element_id, entity_name):
    """
    Create a new entity node related to the specified chunk.
    - Label: entity_name (dynamic)
    - Properties: name = entity_name
    - Relationship: (entity)-[:FROM_CHUNK]->(chunk)
    """
    with driver.session() as session:
        session.run(
            f"""
            CREATE (n:`{entity_name}` {{name: $name}})
            WITH n
            MATCH (c)
            WHERE elementId(c) = $element_id
            CREATE (n)-[:FROM_CHUNK]->(c)
            """,
            name=entity_name, element_id=chunk_element_id
        )
        print(f"Created entity node '{entity_name}' linked to Chunk with elementId {chunk_element_id}.")

def create_relationship(driver, node1_name, relation_type, node2_name):
    """
    Establish a relationship between two nodes by their names.
    - Assumes nodes have a 'name' property.
    - Relationship: (node1)-[:relation_type]->(node2)
    """
    with driver.session() as session:
        session.run(
            f"""
            MATCH (n1 {{name: $name1}}), (n2 {{name: $name2}})
            CREATE (n1)-[:`{relation_type}`]->(n2)
            """,
            name1=node1_name, name2=node2_name
        )
        print(f"Created relationship: '{node1_name}' -[:{relation_type}]-> '{node2_name}'.")

def create_document(driver, doc_type, path):
    """
    Create a new Document node.
    - Properties: document_type, path
    - Returns: elementId of the created document
    """
    with driver.session() as session:
        result = session.run(
            """
            CREATE (d:Document {document_type: $doc_type, path: $path})
            RETURN elementId(d) AS id
            """,
            doc_type=doc_type, path=path
        )
        record = result.single()
        document_element_id = record["id"]
        print(f"Created Document node with elementId {document_element_id}.")
        return document_element_id

def create_document_chunk(driver, document_element_id, index, text, embedding):
    """
    Create a new Chunk node linked to the specified document.
    - Properties: index, text, embedding
    - Relationship: (chunk)-[:FROM_DOCUMENT]->(document)
    - Returns: elementId of the created chunk
    """
    with driver.session() as session:
        result = session.run(
            """
            CREATE (c:Chunk {index: $index, text: $text, embedding: $embedding})
            WITH c
            MATCH (d)
            WHERE elementId(d) = $document_id
            CREATE (c)-[:FROM_DOCUMENT]->(d)
            RETURN elementId(c) AS id
            """,
            index=index, text=text, embedding=embedding, document_id=document_element_id
        )
        record = result.single()
        chunk_element_id = record["id"]
        print(f"Created Chunk node with elementId {chunk_element_id} linked to Document {document_element_id}.")
        return chunk_element_id

def main():
    driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)
    
    # Define the embedding list (truncated in the query, but using provided values)
    embedding = [
        0.011327064596116543, 0.023420892655849457, 0.03523365035653114,
        0.005819568410515785, 0.03588000312447548, -0.030048348009586334,
        0.00917533878237009  # Add more values if available, or use as is
    ]
    
    try:
        # Create the document
        document_element_id = create_document(driver, "inline_text", "document.txt")
        
        # Create the chunk linked to the document
        chunk_element_id = create_document_chunk(driver, document_element_id, 0, "Loren Ipsum Hello world (+ 50 words here)", embedding)
        
        # Example usage of knowledge graph functions
        create_entity_node(driver, chunk_element_id, "Genghis Khan")
        create_entity_node(driver, chunk_element_id, "Lady Jessica")
        create_relationship(driver, "Genghis Khan", "MENTORED_BY", "Lady Jessica")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.close()
        print("Driver closed.")

if __name__ == "__main__":
    main()