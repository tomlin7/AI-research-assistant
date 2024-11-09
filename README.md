# Semantic Document Search System with pgvector & [pgai](https://github.com/timescale/pgai)

*This is a submission for the [Open Source AI Challenge with pgai and Ollama](https://dev.to/challenges/pgai)*

## What I Built

This is a semantic document search for smart document storage and retrieval using natural language queries. You can use natural language to fetch data stored in the PostgreSQL database. It uses pgvector for vector similarity search, pgai through TimescaleDB for AI features.

**Key Features:**
- Semantic search capability using document embeddings
- AI capabilities powered by [pgai](https://github.com/timescale/pgai)
- User-friendly interface built with Streamlit
- Document addition and indexing from GUI
- Batch document processing (directly upload CSV files)
- Rich metadata support for categorization
- Simple table view and a detailed view for data
- Scalable vector search using pgvector's IVFFlat indexing

In cases where you have to manage and search through large collections of documents based on meaning rather than just keywords, this tool is very helpful. Particularly useful for knowledge bases, content management systems, etc.

### Demo
Demo website hosted in Streamlit community cloud, [**visit here**](https://semantic-document-search.streamlit.app)

![image](https://github.com/user-attachments/assets/ecad7f26-ac7c-4aff-8a2e-d6ad44ba406a)

### Features Showcase:

1. **Document Addition**
   - Simple form on nav bar for document addition
   - Support for metadata: category, difficulty level
   - Real-time embedding generation and storage

2. **Semantic Search**
   - Natural language query support, no need to write sql queries!
   - Similarity scores for search results
   - Configurable number of results
   - Expandable result cards (see content and metadata)

3. **Document Management**
   - Dual view options: Table and Detailed view
   - Chronological organization
   - Rich metadata display
   - Clean, intuitive interface

### Tools Used

#### PostgreSQL + pgvector + pgai + Streamlit
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
3. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   # or if you have poetry
   poetry install && poetry shell
   ```

4. **Run the Application**
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

This project is about integrating AI vector search features with traditional databases (which are hard to get used to). This is a very helpful tool for content management systems where you need to manage and search through large collections of documents. Integration of pgvector and pgai provides a great solution along with Streamlit for simple and user-friendly interfaces.

### TODO

- [ ] Better visualization of results using charts and stuff
- [x] Batch document processing (import CSV)
- [ ] Delete, update documents functionality
- [ ] Filtering based on metadata as well
- [ ] More use cases of pgai
