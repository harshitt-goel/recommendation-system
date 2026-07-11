import os
import pickle
import shutil


os.makedirs("models", exist_ok=True)


src = "models/mf_bce_best.pth"
dst = "models/model_weights.pt"

if not os.path.exists(src):
    raise FileNotFoundError(f"{src} not found. Train the model first.")

shutil.copy(src, dst)
print("✓ Copied model weights -> models/model_weights.pt")

model_config = {
    "num_users": num_users,
    "num_items": num_items,
    "embedding_dim": 32,
}

with open("models/model_config.pkl", "wb") as f:
    pickle.dump(model_config, f)

print("✓ Saved model_config.pkl")

artifacts = {
    "user_to_idx": user_to_idx,
    "item_to_idx": item_to_idx,
    "idx_to_item": idx_to_item,
    "user_seen_items": user_seen_items,
    "popular_products": popular_products,
}

with open("models/artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("✓ Saved artifacts.pkl")

print("\n✅ Everything is ready for the FastAPI app.")