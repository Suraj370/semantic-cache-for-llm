from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api import feedback_router, invalidation_router, router, tuner_router
from src.api.dependencies import _vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up singletons on startup so the first request isn't slow
    _vector_store()
    yield


app = FastAPI(
    title="Semantic Cache",
    description="OpenAI-compatible semantic cache proxy. "
                "Change your base_url and get automatic caching — zero code changes.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
app.include_router(invalidation_router)
app.include_router(tuner_router)
app.include_router(feedback_router)


@app.get("/health")
def health():
    store = _vector_store()
    return {"status": "ok", "cache_size": store.size()}
