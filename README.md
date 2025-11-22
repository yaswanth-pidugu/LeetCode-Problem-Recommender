# LeetCode Semantic Problem Recommender

A learning-to-rank recommendation system that recommends the next best LeetCode problem using semantic embeddings and a supervised ranking model. The system uses Sentence-BERT for contextual understanding and LightGBM for ordering problem relevance.

**Live Demo:** https://leetcode-problem-recommender.streamlit.app/

---

## Key Features

### Semantic Retrieval (Sentence-BERT)
Generates dense embeddings using `all-MiniLM-L6-v2` to capture conceptual similarity between problems, enabling retrieval beyond keyword matching.

### Feature Engineering
Each candidate problem is ranked using the following signals:

- **Cosine Similarity:** Measures semantic closeness.
- **Jaccard Similarity:** Measures overlap in problem tags.
- **Difficulty Distance:** Penalizes sudden jumps in difficulty.
- **Popularity Score:** Based on likes, acceptance rate, and submission statistics.

### Learning-to-Rank with LightGBM
A LightGBM ranking model combines all features into a single relevance score, improving recommendation accuracy compared to similarity-only approaches.

---

## Tech Stack

- **Frontend:** Streamlit  
- **Model Serving:** LightGBM, Scikit-Learn  
- **Embeddings:** Sentence-Transformers (HuggingFace)  
- **Preprocessing:** Pandas, NumPy, NLTK  
- **Similarity Metrics:** NumPy cosine similarity  

---

## Project Structure



```bash
├── app/
│   ├── recommender.py    # Core inference logic & feature extraction
│   └── __init__.py
├── data/
│   ├── preprocessed_data.csv  # Cleaned LeetCode dataset
│   └── sbert_embeddings.pkl   # Pre-computed vector embeddings (Cached)
├── models/
│   └── lightgbm_model.pkl     # Trained Ranking Model
├── Data_Pipeline/             # Jupyter Notebooks for EDA & Training
│   ├── leetcode_scraper.ipynb
│   ├── data_preprocessing.ipynb
│   └── lightGBM.ipynb
├── app.py                # Streamlit Application Entry Point
├── requirements.txt      # Python Dependencies
└── README.md             # Documentation


