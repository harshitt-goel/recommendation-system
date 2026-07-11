import torch
import pickle

from app.model import MatrixFactorization


class Recommender:
    """
    Loads trained model weights + artifacts once at startup.
    Serves recommendations via dot-product scoring over all items.
    Falls back to popularity ranking for unknown/cold-start users.
    """

    def __init__(self, models_dir: str = "models"):
        print("Loading model weights and artifacts...")

        # Load config and rebuild model architecture
        with open(f"{models_dir}/model_config.pkl", "rb") as f:
            config = pickle.load(f)

        self.model = MatrixFactorization(
            num_users=config["num_users"],
            num_items=config["num_items"],
            embedding_dim=config["embedding_dim"],
        )
        self.model.load_state_dict(
            torch.load(f"{models_dir}/model_weights.pt", map_location="cpu")
        )
        self.model.eval()

        # Load lookup dicts
        with open(f"{models_dir}/artifacts.pkl", "rb") as f:
            artifacts = pickle.load(f)

        self.user_to_idx      = artifacts["user_to_idx"]
        self.item_to_idx      = artifacts["item_to_idx"]
        self.idx_to_item      = artifacts["idx_to_item"]
        self.user_seen_items  = artifacts["user_seen_items"]
        self.popular_products = artifacts["popular_products"]

        self.num_items = config["num_items"]
        print(f"Model ready. {config['num_users']} users, {config['num_items']} items.")

    def recommend(self, user_id: int, k: int = 10) -> dict:
        """
        Returns top-k recommendations for a given raw user ID.
        Falls back to popularity for unknown users (cold-start).
        """

        # --- Cold-start fallback ---
        if user_id not in self.user_to_idx:
            popular_items = self.popular_products[:k]
            return {
                "user_id": user_id,
                "strategy": "popularity_fallback",
                "recommendations": [int(item) for item in popular_items],
            }

        # --- MF-based personalized recommendations ---
        user_idx = self.user_to_idx[user_id]

        with torch.no_grad():
            user_vector = self.model.user_embedding.weight[user_idx]
            scores = torch.matmul(self.model.item_embedding.weight, user_vector)

        # Mask out already-seen items
        seen = self.user_seen_items.get(user_idx, set())
        if seen:
            scores[list(seen)] = -1e9

        top_indices = torch.topk(scores, k).indices.numpy()
        recommended_item_ids = [int(self.idx_to_item[idx]) for idx in top_indices]

        return {
            "user_id": user_id,
            "strategy": "matrix_factorization",
            "recommendations": recommended_item_ids,
        }
