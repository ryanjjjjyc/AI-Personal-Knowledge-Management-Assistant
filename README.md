# Project setup

## 1. Install Ollama

See https://ollama.com/download for your OS.

```bash
# Pull the models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
# If your machine has less RAM, use qwen3.5:0.8b instead.
```

## 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
```
## 3. Core LangChain + RAG dependencies

```bash
pip install ollama langchain langchain-core langchain-text-splitters langchain-chroma langchain-ollama pypdf
pip install -U langchain
```

## 4. FastAPI backend dependencies

```bash
pip install "fastapi" "uvicorn[standard]" "python-multipart"
```

## 5. Auth dependencies

```bash
pip install "python-jose" "bcrypt"
```

## 6. Environment loading

```bash
pip install python-dotenv
```