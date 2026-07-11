# 🛒 Recommendation System using Matrix Factorization

An end-to-end recommendation system built using **PyTorch**, **FastAPI**, and **Docker** that generates personalized product recommendations from implicit user interactions.

The project follows a production-style ML workflow—from data preprocessing and model training to API deployment and containerization.

---

## 🚀 Features

- Personalized product recommendations using **Matrix Factorization**
- Implicit feedback learning from user interactions
- Weighted interaction scoring (View, Add-to-Cart, Purchase)
- Negative sampling for efficient training
- Cold-start handling using popularity-based recommendations
- FastAPI REST API for inference
- Dockerized deployment
- Model serialization and loading
- Interactive Swagger API documentation

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| ML Framework | PyTorch |
| Backend | FastAPI |
| Data Processing | Pandas, NumPy |
| Deployment | Docker |
| API Docs | Swagger UI |

---

# 📂 Project Structure

```text
recommendation-system/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   └── recommender.py
│
├── models/
│   ├── artifacts.pkl
│   ├── model_config.pkl
│   └── model_weights.pt
│
├── notebooks/
│   └── recommendation_system_clean.ipynb
│
├── Dockerfile
├── requirements.txt
├── save_model.py
└── README.md
```

---

# 🧠 Recommendation Model

The recommendation engine uses **Matrix Factorization**, where each user and product is represented by a learnable embedding vector.

The predicted preference score is computed as the dot product of the corresponding user and item embeddings.

### Training Pipeline

- Data preprocessing
- User & item encoding
- Weighted implicit feedback generation
- Negative sampling
- Matrix Factorization training
- Recall@K evaluation
- Model serialization
- FastAPI deployment

---

# 🌐 API Endpoints

## Health Check

```http
GET /health
```

---

## Get Recommendations

```http
GET /recommend/{user_id}
```

Example:

```http
GET /recommend/123?k=10
```

---

## Recommendation Metadata

```http
GET /recommend/{user_id}/similar
```

---

# 🐳 Running with Docker

## Build

```bash
docker build -t recsys-api .
```

## Run

```bash
docker run -p 8000:8000 recsys-api
```

API Documentation:

```
http://localhost:8000/docs
```

---

# 💻 Running Locally

Create a virtual environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the API

```bash
uvicorn app.main:app --reload
```

Open

```
http://127.0.0.1:8000/docs
```

---

# 📊 Example Response

```json
{
  "user_id": 123,
  "strategy": "matrix_factorization",
  "recommendations": [
    51234,
    21987,
    87124,
    65231,
    44192
  ]
}
```

---

# 📈 Evaluation

The model was evaluated using **Recall@K**, a commonly used ranking metric for recommender systems.

---

# 📦 Dataset

This project uses the **RetailRocket E-Commerce Dataset**.

Download it from Kaggle:

https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

Place the downloaded files inside:

```text
data/raw/
```

The dataset is not included in this repository due to its large size.

---

# 🔮 Future Improvements

- Neural Collaborative Filtering (NCF)
- Two-Tower Retrieval Architecture
- FAISS-based Approximate Nearest Neighbor Search
- Real-time Recommendation Pipeline
- Redis Caching
- Online Recommendation Evaluation

---

# 👨‍💻 Author

**Harshit Goel**

- GitHub: https://github.com/harshitt-goel
- LinkedIn: https://www.linkedin.com/in/harshitgoel22/

---

## ⭐ If you found this project useful, consider giving it a star!
