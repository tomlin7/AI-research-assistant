import os

from document_search.core import DocumentSearchSystem

# Self hosted
# db_params = {
#     'dbname': 'vectordb',
#     'user': 'postgres',
#     'password': 'your_password',
#     'host': 'localhost',
#     'port': '5432'
# }

# TimescaleDB 
db_params = os.getenv("PSQL_URL")

search_system = DocumentSearchSystem(db_params)

docs = [
    {
        'title': 'Python Programming',
        'content': 'Python is a high-level programming language known for its simplicity and readability.',
        'metadata': {'category': 'programming', 'level': 'beginner'}
    },
    {
        'title': 'Machine Learning',
        'content': 'Machine learning is a subset of artificial intelligence that focuses on data and algorithms.',
        'metadata': {'category': 'ai', 'level': 'intermediate'}
    }
]
for doc in docs:
    search_system.add_document(doc['title'], doc['content'], doc['metadata'])

query = "What is AI and machine learning?"
results = search_system.semantic_search(query, limit=2)

print(f"\nSearch results for: {query}")
for result in results:
    print(f"\nTitle: {result['title']}")
    print(f"Content: {result['content']}")
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Metadata: {result['metadata']}")

search_system.close()
