from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from app.recommender import Recommender

# Global recommender instance - loaded ONCE at startup, reused for every request
recommender: Recommender = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, clean up at shutdown."""
    global recommender
    recommender = Recommender(models_dir="models")
    yield
    # (cleanup if needed)


app = FastAPI(
    title="E-Commerce Recommendation API",
    description="Personalized product recommendations using Matrix Factorization trained on implicit user-item interactions.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Health check endpoint — confirms API is running and model is loaded."""
    return {"status": "ok", "model_loaded": recommender is not None}


@app.get("/recommend/{user_id}")
def recommend(
    user_id: int,
    k: int = Query(default=10, ge=1, le=50, description="Number of recommendations to return"),
):
    """
    Get top-k product recommendations for a user.

    - **user_id**: raw visitor ID from the original dataset
    - **k**: number of recommendations (1-50, default 10)

    For users not seen during training (cold-start), falls back to popularity-based recommendations.
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        result = recommender.recommend(user_id=user_id, k=k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/{user_id}/similar")
def recommend_with_meta(user_id: int, k: int = Query(default=10, ge=1, le=50)):
    """
    Same as /recommend but also returns metadata about the request.
    Useful for debugging / interview demo.
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    result = recommender.recommend(user_id=user_id, k=k)
    result["num_items_in_catalog"] = recommender.num_items
    result["seen_items_masked"] = user_id in recommender.user_to_idx
    return result
