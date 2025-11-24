import subprocess

# Run du -sk on the Neo4j data directory inside Docker
result = subprocess.run(['docker', 'exec', 'neo4j-apoc', 'du', '-sk', '/var/lib/neo4j/data'], capture_output=True, text=True)
if result.returncode == 0:
    size_kb = int(result.stdout.strip().split()[0])
    print(f"Current Neo4j database size: {size_kb} KB")
else:
    print("Failed to get database size:", result.stderr)
