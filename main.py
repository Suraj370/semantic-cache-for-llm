from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api import feedback_router, invalidation_router, monitoring_router, router, tuner_router
from src.api.dependencies import _vector_store
import monitoring.metrics as m


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
app.include_router(monitoring_router)


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    """Prometheus scrape endpoint. Point your prometheus.yml here."""
    m.update_cache_size(_vector_store().size())
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    store = _vector_store()
    return {
        "status": "ok",
        "cache_size": store.size(),
        "evictions": store.eviction_count,
        **{k: v for k, v in m.summary().items() if k in ("hits", "misses", "hit_rate", "cost_saved_usd")},
    }
