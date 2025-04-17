# Sample RAG Service

This project implements a Retrieval-Augmented Generation (RAG) service using FastAPI and Python. It processes PDF files containing medical codes and names, stores them in a vector database, and provides an API for querying the data.

## Features
- Extracts medical codes and names from PDF files in the `docs/` directory.
- Uses SentenceTransformers for embedding generation.
- Stores embeddings in a FAISS vector database in the `rag/` directory.
- Provides a FastAPI endpoint for querying the data.
- Includes API key authentication for secure access.

## Project Structure
- `main.py`: Entry point for the FastAPI application.
- `routers/`: Contains API route definitions.
- `controllers/`: Handles business logic for API endpoints.
- `services/`: Contains the RAG service implementation.
- `middlewares/`: Includes middleware for API key authentication.
- `utils/`: Utility functions for initialization and setup.

## Environment Variables
- `API_KEY`: The API key for authenticating requests. Add this to a `.env` file in the root directory.

## Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd sample-rag
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create the required directories:
   ```bash
   mkdir docs rag
   ```

4. Add a `.env` file with the following content:
   ```env
   API_KEY=your_api_key_here
   ```

5. Add PDF files to the `docs/` directory.

6. Run the application:
   ```bash
   uvicorn main:app --reload
   ```

## API Endpoints
- `GET /query`: Query the RAG service with a question.
  - Headers: `X-API-Key: <your_api_key>`
  - Query Parameters: `question=<your_question>`

## Notes
- The `rag/` directory stores the FAISS index and is ignored by version control.
- Ensure the `docs/` directory contains valid PDF files for processing.
