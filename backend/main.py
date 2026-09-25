from dotenv import load_dotenv
load_dotenv()  # must run before importing auth

from contextlib import asynccontextmanager
from pathlib import Path
import threading

from fastapi import (
    Depends, FastAPI, File, HTTPException, UploadFile, status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from RAG import qa_agent
from RAG.qa_agent import clean_answer
from backend.auth import authenticate_admin, create_access_token, get_current_admin

ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


class AppState:
    vector_store = None
    agent = None


state = AppState()
index_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build once at startup (reuses ./db if present)
    state.vector_store = qa_agent.get_vector_store()
    state.agent = qa_agent.build_agent(state.vector_store)
    yield


app = FastAPI(title="RAG API", lifespan=lifespan)


# ---------- Schemas ----------

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


class UploadResponse(BaseModel):
    filename: str
    size: int
    reindexed: bool


# ---------- Auth ----------

@app.post("/auth/token", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    if not authenticate_admin(form.username, form.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({"sub": form.username, "role": "admin"})
    return Token(access_token=token)


# ---------- Upload (admin only) ----------

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    admin: dict = Depends(get_current_admin),
):
    if not file.filename:
        raise HTTPException(400, "Missing filename")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large")

    docs_dir = Path(qa_agent.DOCS_DIR)
    docs_dir.mkdir(parents=True, exist_ok=True)
    dest = docs_dir / Path(file.filename).name  # strip any dir traversal
    dest.write_bytes(content)

    # Rebuild the index in a worker thread so we don't block the event loop.
    def _rebuild():
        with index_lock:
            # add only the new file to the live store
            n = qa_agent.index_file(state.vector_store, dest)
            print(f"Indexed {n} chunks from {dest.name}")
            # no need to rebuild the agent:
            # the middleware holds the same vector_store object

    try:
        await run_in_threadpool(_rebuild)
    except Exception as e:
        raise HTTPException(500, f"Index rebuild failed: {e}")

    return UploadResponse(filename=dest.name, size=len(content), reindexed=True)


# ---------- Query ----------

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    # Sync def -> FastAPI runs this in a threadpool automatically.
    agent = state.agent
    if agent is None:
        raise HTTPException(503, "Agent not ready")

    result = agent.invoke({
        "messages": [{"role": "user", "content": req.question}],
        "context": [],
    })

    answer = clean_answer(result["messages"][-1].content)
    sources = sorted({
        d.metadata.get("source", "unknown")
        for d in result.get("context", [])
    })
    return QueryResponse(answer=answer, sources=sources)