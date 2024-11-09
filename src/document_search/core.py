import json
import textwrap
from typing import Any, Dict, List

import psycopg2
import streamlit as st
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer


class DocumentSearchSystem:
    """Semantic Document Search System with pgvector PGAI

    This is a semantic document search for smart document storage and retrieval using natural language queries. 
    You can use natural language to fetch data stored in the PostgreSQL database. It uses pgvector for vector 
    similarity search, pgai through TimescaleDB for AI features.

    Key Features:
    - Semantic search capability using document embeddings
    - AI capabilities powered by [pgai](https://github.com/timescale/pgai)
    - User-friendly interface built with Streamlit
    - Document addition and indexing from GUI
    - Rich metadata support for categorization
    - Simple table view and a detailed view for data
    - Scalable vector search using pgvector's IVFFlat indexing

    In cases where you have to manage and search through large collections of documents based on meaning rather 
    than just keywords, this tool is very helpful. Particularly useful for knowledge bases, content management 
    systems, etc."""
    
    def __init__(self, db_params: str | Dict[str, Any]):
        """
        Args:
            db_params: parameters for psycopg2.connect
                can be the timescaleDB URL, or the local postgreSQL credentials
        """

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
            try:
                st.error(f"Database setup error: {e}")
            except:
                print(f"Database setup error: {e}")
            raise

    def add_document(self, title: str, content: str, metadata: Dict = None):
        try:
            embedding = self.model.encode(content)
            metadata_json = json.dumps(metadata) if metadata else '{}'
            
            with self.connection.cursor() as cursor:
                cursor.execute('''
                    INSERT INTO documents (title, content, embedding, metadata)
                    VALUES (%s, %s, %s::vector, %s::jsonb)
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
            try:
                st.error(f"Search error: {e}")
            except:
                print(f"Search error: {e}")
            raise
    
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute('''
                    SELECT id, title, content, metadata, created_at
                    FROM documents
                    ORDER BY created_at DESC;
                ''')
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'id': row[0],
                        'title': row[1],
                        'content': row[2],
                        'metadata': row[3],
                        'created_at': row[4]
                    })
                
                return results
        except Exception as e:
            st.error(f"Error retrieving documents: {e}")
            raise

    def close(self):
        if self.connection:
            self.connection.close()