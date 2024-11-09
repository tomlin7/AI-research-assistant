import os
import streamlit as st
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import json
import pandas as pd
from dotenv import load_dotenv
import ollama
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

load_dotenv()

class EnhancedResearchAssistant:
    def __init__(self, db_params: str | Dict[str, Any]):
        """Initialize the research assistant with database parameters."""
        self.db_params = db_params
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.connection = None
        self.ollama_model = "mistral"  # llama2, codellama, etc.
        self.setup_database()

    def setup_database(self):
        """Set up the database with necessary extensions and tables."""
        try:
            self.connection = psycopg2.connect(self.db_params)
            with self.connection.cursor() as cursor:
                cursor.execute('CREATE EXTENSION IF NOT EXISTS vector;')
                cursor.execute('CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;')
                cursor.execute('CREATE EXTENSION IF NOT EXISTS ai CASCADE;')
                
                register_vector(self.connection)
                # cursor.execute('DROP TABLE documents;')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS documents (
                        id SERIAL PRIMARY KEY,
                        title TEXT,
                        content TEXT,
                        embedding VECTOR(384),
                        summary TEXT,
                        key_points JSONB DEFAULT '[]'::JSONB,
                        sentiment FLOAT,
                        metadata JSONB DEFAULT '{}'::JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_analyzed TIMESTAMP,
                        related_topics JSONB DEFAULT '[]'::JSONB
                    );
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS documents_embedding_vs_idx 
                    ON documents 
                    USING diskann (embedding vector_cosine_ops)
                    WITH (num_dimensions = 384);
                ''')

                # cursor.execute('''
                #     CREATE INDEX documents_embedding_idx 
                #     ON documents 
                #     USING ivfflat (embedding vector_cosine_ops)
                #     WITH (lists = 100);
                # ''')
                
                self.connection.commit()
        except Exception as e:
            st.error(f"Database setup error: {e}")
            raise

    def analyze_with_ollama(self, text: str, prompt: str) -> str:
        """Use Ollama Python client to analyze text."""
        try:
            full_prompt = f"{prompt}\n\nText: {text}"
            response = ollama.generate(
                model=self.ollama_model,
                prompt=full_prompt,
                stream=False
            )
            
            return response['response']
        except Exception as e:
            st.error(f"Ollama analysis error: {str(e)}")
            return ""

    def get_sentiment_from_text(self, text: str) -> float:
        """Get sentiment score using Ollama."""
        prompt = (
            "Analyze the sentiment of this text and return only a number between "
            "-1 (very negative) and 1 (very positive). Return only the number, "
            "no other text."
        )
        result = self.analyze_with_ollama(text, prompt)
        try:
            # Clean the result and convert to float
            cleaned_result = result.strip().split()[0]  # Take first word only
            return float(cleaned_result)
        except (ValueError, IndexError):
            st.warning("Could not parse sentiment score, defaulting to 0")
            return 0.0

    def extract_key_points(self, text: str) -> List[str]:
        """Extract key points from text using Ollama."""
        prompt = (
            "Extract 3-5 key points from this text. Format your response as a "
            "Python list of strings. Example format: ['point 1', 'point 2', 'point 3']. "
            "Return only the list, no other text."
        )
        result = self.analyze_with_ollama(text, prompt)
        try:
            # Clean and evaluate the string as a Python list
            cleaned = result.strip().replace("```python", "").replace("```", "")
            return eval(cleaned) if cleaned.startswith("[") else [result]
        except Exception as e:
            st.warning(f"Could not parse key points, using raw response: {str(e)}")
            return [result]

    def generate_summary(self, text: str) -> str:
        """Generate a concise summary using Ollama."""
        prompt = (
            "Generate a concise 2-3 sentence summary of this text. "
            "Focus on the main points and key takeaways. "
            "Make it clear and easy to understand."
        )
        return self.analyze_with_ollama(text, prompt)
    
    def suggest_related_topics(self, text: str) -> List[str]:
        """Suggest related topics or research directions."""
        prompt = (
            "Based on this text, suggest 3-4 related topics or research directions "
            "that might be interesting to explore. Format as a Python list of strings. "
            "Be specific but concise."
        )
        result = self.analyze_with_ollama(text, prompt)
        try:
            cleaned = result.strip().replace("```python", "").replace("```", "")
            return eval(cleaned) if cleaned.startswith("[") else [result]
        except:
            return [result]

    def add_document(self, title: str, content: str, metadata: Dict = None) -> int:
        """Add a document with enhanced analysis."""
        try:
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            
            status_text.text("Generating document embedding...")
            embedding = self.model.encode(content)
            progress_bar.progress(0.2)
            
            status_text.text("Generating summary...")
            summary = self.generate_summary(content)
            progress_bar.progress(0.4)
            
            status_text.text("Extracting key points...")
            key_points = self.extract_key_points(content)
            key_points = json.dumps(key_points) if key_points else '{}'
            progress_bar.progress(0.6)
            
            status_text.text("Analyzing sentiment...")
            sentiment = self.get_sentiment_from_text(content)
            progress_bar.progress(0.8)
            
            status_text.text("Finding related topics...")
            related_topics = self.suggest_related_topics(content)
            related_topics = json.dumps(related_topics) if related_topics else '{}'
            progress_bar.progress(0.9)
            
            metadata_json = json.dumps(metadata) if metadata else '{}'
            
            # Save to database
            status_text.text("Saving to database...")
            with self.connection.cursor() as cursor:
                cursor.execute('''
                    INSERT INTO documents 
                    (title, content, embedding, summary, key_points, sentiment, 
                     metadata, last_analyzed, related_topics)
                    VALUES (%s, %s, %s::vector, %s, %s::jsonb, %s, %s::jsonb, 
                           CURRENT_TIMESTAMP, %s::jsonb)
                    RETURNING id;
                ''', (title, content, embedding.tolist(), summary, key_points, 
                     sentiment, metadata_json, related_topics))
                
                doc_id = cursor.fetchone()[0]
                self.connection.commit()
                
            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")
            return doc_id
            
        except Exception as e:
            st.error(f"Error adding document: {str(e)}")
            self.connection.rollback()
            raise
        finally:
            # Clean up progress indicators
            try:
                progress_bar.empty()
                status_text.empty()
            except:
                pass

    def semantic_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Enhanced semantic search with insights."""
        try:
            query_embedding = self.model.encode(query)
            
            with self.connection.cursor() as cursor:
                cursor.execute('''
                    SELECT 
                        id, title, content, summary, key_points, sentiment,
                        metadata, 1 - (embedding <=> %s::vector) as similarity
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
                        'summary': row[3],
                        'key_points': row[4],
                        'sentiment': row[5],
                        'metadata': row[6],
                        'similarity': float(row[7])
                    })
                
                return results
        except Exception as e:
            st.error(f"Search error: {e}")
            raise

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents with their insights."""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute('''
                    SELECT 
                        id, title, content, summary, key_points, sentiment,
                        metadata, created_at, last_analyzed, related_topics
                    FROM documents
                    ORDER BY created_at DESC;
                ''')
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'id': row[0],
                        'title': row[1],
                        'content': row[2],
                        'summary': row[3],
                        'key_points': row[4],
                        'sentiment': row[5],
                        'metadata': row[6],
                        'created_at': row[7],
                        'last_analyzed': row[8],
                        'related_topics': row[9]
                    })
                
                return results
        except Exception as e:
            st.error(f"Error retrieving documents: {e}")
            raise

    def generate_insights_visualization(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate visualizations from document collection."""
        if not documents:
            return {}
        
        # Prepare data for visualizations
        sentiments = [doc['sentiment'] for doc in documents]
        titles = [doc['title'] for doc in documents]
        categories = [doc['metadata'].get('category', 'uncategorized') for doc in documents]
        
        # Sentiment timeline
        fig_sentiment = go.Figure(data=[
            go.Bar(x=titles, y=sentiments, marker_color='lightblue')
        ])
        fig_sentiment.update_layout(
            title="Document Sentiment Analysis",
            xaxis_title="Documents",
            yaxis_title="Sentiment Score"
        )
        
        # Category distribution
        category_counts = pd.Series(categories).value_counts()
        fig_categories = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Document Categories Distribution"
        )
        
        # Generate word cloud
        all_text = " ".join([doc['content'] for doc in documents])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        
        # Convert word cloud to base64 image
        img_buffer = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(img_buffer, format='PNG', bbox_inches='tight', pad_inches=0)
        img_buffer.seek(0)
        wordcloud_b64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        return {
            'sentiment_plot': fig_sentiment,
            'category_plot': fig_categories,
            'wordcloud': wordcloud_b64
        }

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()

def main():
    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 AI Research Assistant")
    
    st.sidebar.title("⚙️ Settings")
    selected_model = st.sidebar.selectbox(
        "Select Ollama Model",
        ["tinyllama", "mistral", "llama2", "codellama", "neural-chat"],
        help="Choose the AI model for analysis"
    )

    db_params = os.getenv("PSQL_URL")
    assistant = EnhancedResearchAssistant(db_params)
    assistant.ollama_model = selected_model

    with st.sidebar.expander("ℹ️ Current Model Info"):
        try:
            model_info = ollama.show(model=selected_model)
            st.json(model_info)
        except Exception as e:
            st.error(f"Could not fetch model info: {str(e)}")

    st.sidebar.title("📄 Add New Document")
    with st.sidebar.form("add_document_form"):
        title = st.text_input("Document Title")
        content = st.text_area("Document Content")
        category = st.selectbox(
            "Category",
            ["Research", "Technology", "Science", "Business", "Literature", "Other"]
        )
        importance = st.slider("Document Importance", 1, 5, 3)
        
        submitted = st.form_submit_button("Add & Analyze Document")
        
        if submitted and title and content:
            metadata = {
                'category': category.lower(),
                'importance': importance
            }
            with st.spinner("Analyzing document with AI..."):
                doc_id = assistant.add_document(title, content, metadata)
                st.sidebar.success(f"Document analyzed and added! ID: {doc_id}")

    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📊 Insights", "📚 Document Library"])
    with tab1:
        st.subheader("Semantic Document Search")
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("Enter your search query")
        with col2:
            num_results = st.number_input("Number of results", 1, 10, 5)

        if search_query:
            results = assistant.semantic_search(search_query, limit=num_results)
            
            if results:
                for idx, result in enumerate(results, 1):
                    with st.expander(
                        f"{idx}. {result['title']} (Similarity: {result['similarity']:.2f})"
                    ):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown("**Summary:**")
                            st.write(result['summary'])
                            
                            st.markdown("**Content:**")
                            st.write(result['content'])
                            
                        with col2:
                            st.markdown("**Key Points:**")
                            for point in result['key_points']:
                                st.markdown(f"• {point}")
                            
                            st.markdown("**Sentiment:**")
                            sentiment = float(result['sentiment'])
                            st.progress((sentiment + 1) / 2)
                            st.write(f"Score: {sentiment:.2f}")
                            
                            st.markdown("**Metadata:**")
                            st.json(result['metadata'])

    with tab2:
        st.subheader("Document Collection Insights")
        documents = assistant.get_all_documents()
        
        if documents:
            visualizations = assistant.generate_insights_visualization(documents)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(visualizations['sentiment_plot'], use_container_width=True)
            with col2:
                st.plotly_chart(visualizations['category_plot'], use_container_width=True)
            
            st.subheader("Document Word Cloud")
            st.image(f"data:image/png;base64,{visualizations['wordcloud']}")
        else:
            st.info("Add some documents to see insights!")

    with tab3:
        st.subheader("Document Library")
        documents = assistant.get_all_documents()
        
        if documents:
            tab1, tab2 = st.tabs(["Compact", "Detailed"])
            with tab1:
                df = pd.DataFrame([{
                    'ID': doc['id'],
                    'Title': doc['title'],
                    'Category': doc['metadata'].get('category', 'N/A'),
                    'Sentiment': f"{float(doc['sentiment']):.2f}",
                    'Created': doc['created_at'].strftime('%Y-%m-%d %H:%M')
                } for doc in documents])
                
                st.dataframe(
                    df,
                    hide_index=True,
                    use_container_width=True
                )
            with tab2:
                for doc in documents:
                    with st.expander(f"📄 {doc['title']} (ID: {doc['id']})"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown("**Summary:**")
                            st.write(doc['summary'])
                            
                            st.markdown("**Content:**")
                            st.write(doc['content'])
                            
                        with col2:
                            st.markdown("**Analysis Details:**")
                            st.markdown("*Key Points:*")

                            try:
                                for point in json.loads(doc['key_points']):
                                    st.markdown(f"- {point}")
                            except:
                                for point in doc['key_points']:
                                    st.markdown(f"- {point}")

                            st.markdown("*Sentiment:*")
                            sentiment = float(doc['sentiment'])
                            st.progress((sentiment + 1) / 2)

                            try:
                                for point in json.loads(doc['related_topics']):
                                    st.markdown(f"- {point}")
                            except:
                                for point in doc['related_topics']:
                                    st.markdown(point)

if __name__ == '__main__':
    main()