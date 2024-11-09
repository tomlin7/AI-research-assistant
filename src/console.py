import json
import os
import textwrap
from typing import Any, Dict, List

import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer


class DocumentSearchSystem:
    def __init__(self, db_params: str | Dict[str, Any]):
        self.db_params = db_params
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.connection = None
        self.setup_database()

    def setup_database(self):
        try:
            self.connection = psycopg2.connect(self.db_params)
            with self.connection.cursor() as cursor:
                cursor.execute('CREATE EXTENSION IF NOT EXISTS vector;')
                cursor.execute('CREATE EXTENSION IF NOT EXISTS ai CASCADE;')
                
                register_vector(self.connection)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS documents (
                        id SERIAL PRIMARY KEY,
                        title TEXT,
                        content TEXT,
                        embedding VECTOR(384), 
                        metadata JSONB DEFAULT '{}'::JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                    ON documents 
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                ''')
                
                self.connection.commit()
        except Exception as e:
            print(f"Database setup error: {e}")
            raise

    def add_document(self, title: str, content: str, metadata: Dict = None):
        try:
            embedding = self.model.encode(content)
            metadata_json = json.dumps(metadata) if metadata else '{}'
            
            with self.connection.cursor() as cursor:
                cursor.execute('''
                    INSERT INTO documents (title, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s::jsonb)
                    RETURNING id;
                ''', (title, content, embedding.tolist(), metadata_json))
                
                doc_id = cursor.fetchone()[0]
                self.connection.commit()
                return doc_id
        except Exception as e:
            print(f"Error adding document: {e}")
            self.connection.rollback()
            raise

    def semantic_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        try:
            query_embedding = self.model.encode(query)
            
            with self.connection.cursor() as cursor:
                cursor.execute('''
                    SELECT 
                        id,
                        title,
                        content,
                        metadata,
                        1 - (embedding <=> %s::vector) as similarity
                    FROM documents
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                ''', (query_embedding.tolist(), query_embedding.tolist(), limit))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'id': row[0],
                        'title': row[1],
                        'content': textwrap.shorten(row[2], width=200),
                        'metadata': row[3],
                        'similarity': float(row[4])
                    })
                
                return results
        except Exception as e:
            print(f"Search error: {e}")
            raise

    def close(self):
        if self.connection:
            self.connection.close()



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
