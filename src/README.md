# Company Knowledge Base Assistant

An intelligent Q&A system that answers questions about company documentation using RAG (Retrieval-Augmented Generation) and MCP (Model Context Protocol) tools.

## Features

- **RAG-powered search**: Semantic search over company documentation using FAISS
- **MCP tools**: Dynamic document reading and management
- **Local LLM**: Privacy-preserving answers using Ollama

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Documents

Create a `docs/` directory and add your company documentation files (`.txt`, `.md`, `.pdf`, `.docx`):

```bash
mkdir docs
# Add your company documentation files here
```

### 3. Configure

Edit `config.py` to set:
- `DOCUMENTS_DIR`: Path to your documentation directory
- `OLLAMA_MODEL`: Local LLM model to use (default: "llama3")
- Other settings as needed

### 4. Build Index (Optional)

The index will be built automatically on first use. To manually build it:

```bash
python main.py build-index
```

Or directly:

```bash
python -m rag.build_index
```

## Usage

### Interactive CLI

Run the interactive assistant:

```bash
python main.py
```

Then ask questions about your company documentation!

## Project Structure

```
src/
├── config.py              # Configuration
├── main.py                # CLI entry point
├── assistant.py           # Main assistant class
├── rag/                   # RAG components
│   ├── ingest.py         # Document ingestion
│   ├── chunk.py          # Text chunking
│   ├── embed.py          # Embedding generation
│   ├── build_index.py    # FAISS index building
│   └── query.py          # Query and retrieval
├── mcp/                   # MCP components
│   ├── server.py         # MCP server with tools
│   └── client.py         # MCP client
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## How It Works

1. **Document Ingestion**: Loads documents from the `docs/` directory
2. **Chunking**: Splits documents into smaller chunks with overlap
3. **Embedding**: Generates embeddings using SentenceTransformers
4. **Indexing**: Builds FAISS vector index for fast similarity search
5. **Query**: 
   - Retrieves relevant chunks using semantic search
   - Optionally uses MCP tools for document access
   - Generates answer using local LLM (Ollama)

## MCP Tools

The MCP server provides:
- `read_document`: Read a specific document
- `list_documents`: List all available documents
- `search_documents`: Search documents by name

## Troubleshooting

**Index not found**: Run `python main.py build-index` first

**Ollama not responding**: Make sure Ollama is running and the model is installed:
```bash
ollama pull llama3
```

**No documents found**: Check that `DOCUMENTS_DIR` in `config.py` points to your documents

## License

MIT
