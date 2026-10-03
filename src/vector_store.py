import faiss
import numpy as np


# -------------------------
# Create FAISS vector store
# -------------------------

def create_vector_store(embeddings):

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# -------------------------
# Search vector store
# -------------------------

def search_vector_store(
    index,
    query_embedding,
    k=3
):

    query_embedding = np.array(
        [query_embedding]
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        k
    )

    return distances, indices


# -------------------------
# Save vector store
# -------------------------

def save_vector_store(
    index,
    file_path
):

    faiss.write_index(
        index,
        file_path
    )


# -------------------------
# Load vector store
# -------------------------

def load_vector_store(
    file_path
):

    index = faiss.read_index(
        file_path
    )

    return index