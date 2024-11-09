import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from document_search import BatchDocumentProcessor, DocumentSearchSystem

load_dotenv() 


def main():
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

    if 'processing_started' not in st.session_state:
        st.session_state.processing_started = False
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = None
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = None
    if 'csv_params' not in st.session_state:
        st.session_state.csv_params = None

    st.set_page_config(
        page_title="Document Search System",
        page_icon="🔍",
        layout="wide"
    )
        
    st.title("📚 Semantic Document Search System")
    search_system = DocumentSearchSystem(db_params)
    batch_processor = BatchDocumentProcessor(search_system)

    tab1, tab2, tab3 = st.tabs(["Search & View", "Add Document", "Batch Upload"])

    with tab1:
        st.subheader("🔍 Search Documents")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("Enter your search query")
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
        display_all_documents(search_system)

    with tab2:
        st.subheader("Add Single Document")
        
        with st.form("add_document_form"):
            title = st.text_input("Document Title")
            content = st.text_area("Document Content")
            category = st.selectbox("Category", ["General", "Technology", "Science", "Business", "Other"])
            level = st.selectbox("Level", ["Beginner", "Intermediate", "Advanced"])
            
            submitted = st.form_submit_button("Add Document")
            
            if submitted and title and content:
                metadata = {
                    'category': category.lower(),
                    'level': level.lower()
                }
                doc_id = search_system.add_document(title, content, metadata)
                st.success(f"Document added successfully! ID: {doc_id}")

    with tab3:
        st.subheader("📥 Batch Document Upload")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # CSV Structure Form
            with st.form("csv_structure_form"):
                st.write("Specify CSV Structure")
                title_col = st.text_input("Title Column Name", "title")
                content_col = st.text_input("Content Column Name", "content")
                metadata_cols = st.multiselect(
                    "Metadata Columns",
                    options=pd.read_csv(temp_file_path).columns.tolist(),
                    default=[]
                )
                batch_size = st.number_input("Batch Size", min_value=10, max_value=1000, value=100)
                max_workers = st.number_input("Max Workers", min_value=1, max_value=8, value=4)
                
                validate_button = st.form_submit_button("Validate CSV")
                
                if validate_button:
                    # Store parameters in session state
                    st.session_state.csv_params = {
                        'file_path': temp_file_path,
                        'title_col': title_col,
                        'content_col': content_col,
                        'metadata_cols': metadata_cols,
                        'batch_size': batch_size,
                        'max_workers': max_workers
                    }
                    # Perform validation
                    st.session_state.validation_results = batch_processor.validate_csv(
                        temp_file_path,
                        title_col,
                        content_col,
                        metadata_cols
                    )
                    st.session_state.processing_complete = False
                    st.session_state.processing_started = False

            # Display validation results if available
            if st.session_state.validation_results:
                if st.session_state.validation_results['valid']:
                    st.success("CSV validation successful!")
                    st.write("Statistics:", st.session_state.validation_results['stats'])
                    
                    # Process Documents button outside of form
                    if not st.session_state.processing_started and not st.session_state.processing_complete:
                        if st.button("Process Documents"):
                            st.session_state.processing_started = True
                            
                    # Handle processing if started
                    if st.session_state.processing_started and not st.session_state.processing_complete:
                        with st.spinner("Processing documents..."):
                            params = st.session_state.csv_params
                            results = batch_processor.process_csv(
                                params['file_path'],
                                params['title_col'],
                                params['content_col'],
                                params['metadata_cols'],
                                params['batch_size'],
                                params['max_workers']
                            )
                            st.session_state.processing_results = results
                            st.session_state.processing_complete = True
                            st.session_state.processing_started = False
                            st.rerun()
                    
                    # Show results if processing is complete
                    if st.session_state.processing_complete and st.session_state.processing_results:
                        results = st.session_state.processing_results
                        st.success("Batch processing completed!")
                        st.write(f"Successfully processed: {results['successful']}")
                        st.write(f"Failed: {results['failed']}")
                        
                        if results['errors']:
                            with st.expander("View Errors"):
                                for error in results['errors']:
                                    st.error(error)
                else:
                    st.error("CSV validation failed!")
                    for error in st.session_state.validation_results['errors']:
                        st.error(error)

            # Cleanup temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    search_system.close()

def display_all_documents(search_system: DocumentSearchSystem):
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

if __name__ == "__main__":
    main()