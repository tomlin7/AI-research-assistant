import json
import os
from typing import Any, Dict, List

import pandas as pd
import psycopg2
import streamlit as st
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv() 


class DocumentSearchSystem:
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
                # documents table with vector search 
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
                
                # ndex for vector similarity search
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                    ON documents 
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                ''')
                
                self.connection.commit()
        except Exception as e:
            st.error(f"Database setup error: {e}")
            raise

    def add_document(self, title: str, content: str, metadata: Dict = None):
        try:
            embedding = self.model.encode(content)
            # metadata to JSON string
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
            st.error(f"Error adding document: {e}")
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
                        'content': row[2],
                        'metadata': row[3],
                        'similarity': float(row[4])
                    })
                
                return results
        except Exception as e:
            st.error(f"Search error: {e}")
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
        """Close the database connection."""
        if self.connection:
            self.connection.close()

# --------------------------------- GUI --------------------------------------

st.set_page_config(
    page_title="Document Search System",
    page_icon="🔍",
    layout="wide"
)

st.title("📚 Semantic Document Search System")

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
st.sidebar.title("Add New Document")

with st.sidebar.form("add_document_form"):
    title = st.text_input("Title")
    content = st.text_area("Content")
    category = st.selectbox("Category", ["General", "Technology", "Science", "Business", "Other"])
    
    # more metadata can be added...
    level = st.selectbox("Level (example metadata field)", ["Beginner", "Intermediate", "Advanced"])
    
    submitted = st.form_submit_button("Add Document")
    
    if submitted and title and content:
        metadata = {
            'category': category.lower(),
            'level': level.lower()
        }
        doc_id = search_system.add_document(title, content, metadata)
        st.sidebar.success(f"Document added successfully! ID: {doc_id}")

st.subheader("🔍 Search Documents")
col1, col2 = st.columns([3, 1])
with col1:
    search_query = st.text_input("Enter your search query (natural language)")
with col2:
    num_results = st.number_input("Number of results", min_value=1, max_value=10, value=5)

if search_query:
    results = search_system.semantic_search(search_query, limit=num_results)
    
    if results:
        st.subheader("Search Results")
        for idx, result in enumerate(results, 1):
            with st.expander(f"{idx}. {result['title']} (Similarity: {result['similarity']:.2f})"):
                st.markdown(f"**Content:**\n{result['content']}")
                st.markdown("**Metadata:**")
                st.json(result['metadata'])
    else:
        st.info("No results found.")

st.subheader("📑 All Documents")
documents = search_system.get_all_documents()
if documents:
    tab1, tab2 = st.tabs(["Table View", "Detailed View"])
    
    with tab1:
        df = pd.DataFrame(documents)
        df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        st.dataframe(
            df[['id', 'title', 'content', 'created_at']],
            hide_index=True,
            column_config={
                "id": st.column_config.NumberColumn("ID", width="small"),
                "title": st.column_config.TextColumn("Title", width="medium"),
                "content": st.column_config.TextColumn("Content", width="large"),
                "created_at": st.column_config.TextColumn("Created At", width="medium"),
            }
        )
    
    with tab2:
        for doc in documents:
            with st.expander(f"📄 {doc['title']} (ID: {doc['id']})"):
                st.markdown("**Content:**")
                st.write(doc['content'])
                st.markdown("**Metadata:**")
                st.json(doc['metadata'])
                st.markdown(f"**Created At:** {doc['created_at']}")
else:
    st.info("No documents in the database yet.")

search_system.close()
