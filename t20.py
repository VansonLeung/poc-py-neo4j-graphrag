import neo4j

# Neo4j connection details (adjust as needed)
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")  # Replace with your actual credentials

def main():
    driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)
    
    try:
        with driver.session() as session:
            # Create the first node
            session.run("CREATE (n:Person {name: $name})", name="Alice")
            print("Created node: Alice")
            
            # Create the second node
            session.run("CREATE (m:Person {name: $name})", name="Bob")
            print("Created node: Bob")
            
            # Create a relationship between the two nodes
            session.run(
                "MATCH (a:Person {name: $name1}), (b:Person {name: $name2}) "
                "CREATE (a)-[:KNOWS]->(b)",
                name1="Alice", name2="Bob"
            )
            print("Created relationship: Alice KNOWS Bob")
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.close()
        print("Driver closed.")

if __name__ == "__main__":
    main()
