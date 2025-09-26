# mini-RAG pipeline

**Tokenization (your tokenize() function)**

```py
def tokenize(text):
    return re.findall(r"\w+", text.lower())
```

- What it does:
  - Splits raw text into lowercase words (["rag","combines","search",...]).
  - No IDs are generated; it just returns a list of strings.

- Real LLM equivalent:
  - A tokenizer would map each piece (token) to an integer ID.

**Embedding Vector Creation (your generate_simple_embedding() function)**

```py
def generate_simple_embedding(text):
    tokens = tokenize(text)
    vector = np.zeros(len(vocab), dtype=float)
    for i, word in enumerate(vocab):
        vector[i] = tokens.count(word)
    return vector
```

- What it does:
  - Builds a vocabulary of all unique words across documents (vocab).
  - Creates a fixed-length vector where each position corresponds to a word in vocab.
  - Each entry is the frequency of that word in the text.
- Real LLM equivalent:
  - Instead of frequency counts, a trained embedding model would output a dense semantic vector (e.g., 1536 floats) for the entire text.

**Storage / Indexing**

```py
embeddings = [generate_simple_embedding(doc) for doc in documents]
```

- Here you’re essentially creating a vector store as a simple Python list.
- Each document’s embedding is stored for later comparison.

**Query Embedding and Retrieval**

```py
query = "how does semantic search work"
embed_query = generate_simple_embedding(query)

similarities = [cos_sim(embed_query, e) for e in embeddings]
top = np.argsort(similarities)[::-1]
```

- The query is tokenized and embedded with the same method as the documents.
- cos_sim() calculates cosine similarity between the query vector and each document vector.
- Documents are ranked by similarity (top).
