import numpy as np
import re


# --- 1) Create vocabulary
def tokenize(text):
    return re.findall(r"\w+", text.lower())


# --- 2) Create vectors
def generate_simple_embedding(text):
    tokens = tokenize(text)
    vector = np.zeros(len(vocab), dtype=float)
    for i, word in enumerate(vocab):
        vector[i] = tokens.count(word)  # word frequency
    return vector


# --- 3) Cosine similarity
def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Example documents
documents = [
    "rag combines search and text generation",
    "embeddings transform text into numerical vectors",
    "python is a great language for data science",
    "semantic search finds texts with similar meaning",
]

vocab = sorted(set(word for doc in documents for word in tokenize(doc)))
print("Vocabulary:", vocab)

embeddings = [generate_simple_embedding(doc) for doc in documents]

query = "how does semantic search work"
embed_query = generate_simple_embedding(query)

# Calculate similarity with each document
similarities = [cos_sim(embed_query, e) for e in embeddings]
top = np.argsort(similarities)[::-1]

print("\nSimilarity ranking:")
for i in top:
    print(f"{similarities[i]:.4f} -> {documents[i]}")
