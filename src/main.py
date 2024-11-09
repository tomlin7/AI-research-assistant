import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from document_search.core import DocumentSearchSystem

load_dotenv() 

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
