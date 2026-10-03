from src.pdf_loader import load_pdf
from src.text_splitter import split_text
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store, search_vector_store


pdf_path = "data/attention-is-all-you-need.pdf"


# 1. Load PDF
text = load_pdf(pdf_path)

# 2. Split PDF into chunks
chunks = split_text(text)

# 3. Create embeddings
embeddings = create_embeddings(chunks)

# 4. Create FAISS vector store
index = create_vector_store(embeddings)

print("Vector store created!")
print("Number of vectors:", index.ntotal)


# 5. Create embedding for user question
query = "What is the Transformer architecture?"

query_embedding = create_embeddings([query])[0]


# 6. Search for similar chunks
distances, indices = search_vector_store(
    index,
    query_embedding,
    k=3
)


# 7. Display results
print("\nTop 3 relevant chunks:\n")

for rank, index_number in enumerate(indices[0], start=1):

    print(f"--- Result {rank} ---")
    print("Distance:", distances[0][rank - 1])
    print(chunks[index_number])
    print()