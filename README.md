# AI Research Assistant with Semantic Document Search System

*This is a submission for the [Open Source AI Challenge with pgai and Ollama](https://dev.to/challenges/pgai)*

## What I Built

This is an AI based research assistant with a semantic document search system for smart document storage and retrieval using natural language queries. [**Ollama**](https://ollama.com/) is integrated into the assistant to summarise, and generate sentiment analysis, key points, related topics for provided content. [Streamlit](https://streamlit.io/) is used to provide a minimalistic user interface.

You can use natural language to search data stored in the PostgreSQL database. Uses **pgvector** for vector similarity search, [**pgai**](https://github.com/timescale/pgai) through **TimescaleDB** for search AI features. It is very helpful in cases where you have to manage and search through large collections of documents based on meaning rather than just keywords.

**Key Features:**
- Uses Ollama to summarise docs, and generate sentiment analysis, key points, and related topics
- Semantic search capability using document embeddings, powered by pgai
- Batch document processing (directly upload CSV files)
- User-friendly interface built with Streamlit
- Document addition and indexing from GUI
- Rich metadata support for categorization
- Simple table view and a detailed view for data
- Scalable vector search using both pgvector's IVFFlat indexing and the [**pgvectorscale**](https://github.com/timescale/pgvectorscale) extension

Although initially the idea was to develop a semantic document search system, later on I decided to extend this to an AI research assistant featuring the same document search system along with Ollama integration.

## Demo

Because of problems with hosting Ollama along with the assistant app, only the semantic document search tool demo is hosted.
- Thanks to Streamlit community cloud, [**visit the demo**](https://semantic-doc-search.streamlit.app) ⭐

![assistant](https://github.com/user-attachments/assets/c58ff3ae-b122-466b-a088-c9e31b80b60f)

![document search tool](https://github.com/user-attachments/assets/ecad7f26-ac7c-4aff-8a2e-d6ad44ba406a)

## Tools Used

### Ollama + pgvector + pgai + Streamlit
- [**Ollama**](https://ollama.com/) is integrated into the assistant to summarise, and generate sentiment analysis, key points, related topics for provided content. 
- [TimescaleDB](https://www.timescale.com/) (PostgreSQL) for primary database (can be configured for self hosted psql as well)
- [pgvector](https://github.com/pgvector/pgvector) for efficient vector similarity search
- [pgai](https://github.com/timescale/pgai) through TimescaleDB for AI
- [Streamlit](https://streamlit.io/) for the web interface

### Key Technologies
1. **Database Layer**
   - pgvector extension for vector operations
   - pgai extension for AI features
   - IVFFlat indexing for efficient similarity search
   - JSONB data type for flexible metadata storage

2. **Machine Learning**
   - [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers) (`all-MiniLM-L6-v2 model`)
   - 384-dimensional embeddings for semantic representation

3. **Backend**
   - Python 3.12+
   - psycopg2 for PostgreSQL interaction
   - Vector similarity calculations using cosine distance

4. **Frontend**
   - Streamlit for the web interface
   - Pandas for data display
   - Download data as CSV files

## Installation

### Using Timescale Cloud

1. **Create a Timescale Service**
   - Open [Timescale Cloud Console](https://console.cloud.timescale.com/) and create a service
   - In the **AI** tab, enable `ai`, `vector` extensions
   - Pick Python app and copy the database connection URL

2. **Configure Environment**
   Edit the `src/.env` file with the copied URL
   ```bash
   PSQL_URL=postgres://username:password@hostname:port/dbname?sslmode=require
   ```
3. Install Ollama and any model (make sure its added to script) for assistant
   ```
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull mistral
   ollama serve
   ```
3. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   # or if you have poetry
   poetry install && poetry shell
   ```

4. **Run the Assistant**
   ```bash
   cd src
   streamlit run assistant.py
   ```
   **Run the Document Search Tool**
   ```bash
   cd src
   streamlit run main.py
   ```

### Self-Hosted PostgreSQL

1. **Install PostgreSQL and Extensions**
   ```bash
   # Install PostgreSQL
   sudo apt-get install postgresql postgresql-common

   # Install pgvector
   sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh
   sudo apt install postgresql-12-pgvector

   # Install pgai
   # https://github.com/timescale/pgai/tree/main?tab=readme-ov-file#install-from-source
   ```

2. **Configure Database**
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   CREATE EXTENSION IF NOT EXISTS ai CASCADE;
   ```

3. **Configure Environment**
   ```bash
   PSQL_URL=postgresql://user:password@localhost:5432/dbname
   ```
   or configure within script
   ```py
    db_params = {
        'dbname': 'dbname',
        'user': 'postgres',
        'password': 'your_password',
        'host': 'localhost',
        'port': '5432'
    }
    ```

## Final Thoughts

This project is about integrating AI vector search features with traditional databases (which are hard to get used to). The same tool is used to create an AI research assistant with Ollama integration. This is a very helpful tool for content management systems where you need to manage and search through large collections of documents. Integration of pgvector and pgai provides a strong solution.

### TODO

- [ ] Better visualization of results using charts and stuff
- [x] Batch document processing (import CSV)
- [ ] Delete, update documents functionality
- [ ] Filtering based on metadata as well
- [ ] More use cases of pgai
