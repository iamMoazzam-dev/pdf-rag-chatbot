import os
import time
import pickle

from dotenv import load_dotenv
from google import genai

from src.pdf_loader import load_pdf
from src.text_splitter import split_text
from src.embeddings import create_embeddings
from src.vector_store import (
    create_vector_store,
    search_vector_store,
    save_vector_store,
    load_vector_store
)


# -------------------------
# Gemini setup
# -------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found. "
        "Please check your .env file."
    )

client = genai.Client(
    api_key=api_key
)


# -------------------------
# Storage locations
# -------------------------

VECTOR_STORE_PATH = "data/faiss.index"
CHUNKS_PATH = "data/chunks.pkl"


# -------------------------
# Load and prepare PDF
# -------------------------

def prepare_pdf(pdf_path):

    print("Loading PDF...")

    pages = load_pdf(pdf_path)

    if not pages:
        raise ValueError(
            "Could not extract usable text from this PDF. "
            "Please upload a text-based PDF."
        )

    print(
        "Number of pages with text:",
        len(pages)
    )

    # -------------------------
    # Split pages into chunks
    # -------------------------

    chunks = split_text(pages)

    if not chunks:
        raise ValueError(
            "No text chunks could be created from this PDF."
        )

    print(
        "Number of chunks:",
        len(chunks)
    )

    # -------------------------
    # Create embeddings
    # -------------------------

    embeddings = create_embeddings(
        chunks
    )

    if len(embeddings) == 0:
        raise ValueError(
            "Could not create embeddings for this PDF."
        )

    # -------------------------
    # Create FAISS index
    # -------------------------

    index = create_vector_store(
        embeddings
    )

    # -------------------------
    # Save FAISS index
    # -------------------------

    save_vector_store(
        index,
        VECTOR_STORE_PATH
    )

    # -------------------------
    # Save chunks
    # -------------------------

    with open(
        CHUNKS_PATH,
        "wb"
    ) as file:

        pickle.dump(
            chunks,
            file
        )

    print("FAISS index saved!")
    print("Chunks saved!")

    return chunks, index


# -------------------------
# Load saved RAG data
# -------------------------

def load_saved_rag():

    if not os.path.exists(
        VECTOR_STORE_PATH
    ):
        return None, None

    if not os.path.exists(
        CHUNKS_PATH
    ):
        return None, None

    # -------------------------
    # Load FAISS index
    # -------------------------

    index = load_vector_store(
        VECTOR_STORE_PATH
    )

    # -------------------------
    # Load chunks
    # -------------------------

    with open(
        CHUNKS_PATH,
        "rb"
    ) as file:

        chunks = pickle.load(
            file
        )

    print(
        "Saved FAISS index loaded!"
    )

    print(
        "Saved chunks loaded!"
    )

    return chunks, index


# -------------------------
# Retrieve relevant chunks
# -------------------------

def retrieve_chunks(
    question,
    index,
    chunks,
    k=3
):

    # -------------------------
    # Create question embedding
    # -------------------------

    query_embedding = create_embeddings(
        [
            {
                "page": 0,
                "text": question
            }
        ]
    )[0]

    # -------------------------
    # Search FAISS
    # -------------------------

    distances, indices = search_vector_store(
        index,
        query_embedding,
        k=k
    )

    # -------------------------
    # Get relevant chunks
    # -------------------------

    retrieved_chunks = []

    for index_number in indices[0]:

        retrieved_chunks.append(
            chunks[index_number]
        )

    return retrieved_chunks


# -------------------------
# Generate answer using Gemini
# -------------------------

def generate_answer(
    question,
    retrieved_chunks
):

    # -------------------------
    # Create context
    # -------------------------

    context_parts = []

    for chunk in retrieved_chunks:

        page_number = chunk["page"]

        text = chunk["text"]

        context_parts.append(
            f"[Page {page_number}]\n{text}"
        )

    context = "\n\n".join(
        context_parts
    )

    # -------------------------
    # Create prompt
    # -------------------------

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using only
the information provided in the context below.

Each section contains a page number.

If the answer cannot be found in the context,
say:

"I could not find the answer in the provided PDF."

At the end of your answer, mention the relevant
PDF page number or numbers.

Context:
{context}

Question:
{question}
"""

    # -------------------------
    # Call Gemini
    # -------------------------

    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )

            return response.text

        except Exception as error:

            error_message = str(error)

            print(
                f"\nGemini attempt {attempt + 1} failed."
            )

            print(
                error_message
            )

            # -------------------------
            # Quota exceeded
            # -------------------------

            if (
                "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
                or "quota" in error_message.lower()
            ):

                return (
                    "Gemini API quota has been exceeded. "
                    "Please try again after your Gemini "
                    "quota resets."
                )

            # -------------------------
            # Temporary server error
            # -------------------------

            if (
                "503" in error_message
                or "UNAVAILABLE" in error_message
            ):

                if attempt < 2:

                    print(
                        "Gemini is temporarily unavailable."
                    )

                    print(
                        "Retrying in 5 seconds..."
                    )

                    time.sleep(5)

                    continue

                return (
                    "Gemini is temporarily unavailable. "
                    "Please try again later."
                )

            # -------------------------
            # Other error
            # -------------------------

            return (
                "An error occurred while generating "
                "the answer. Please try again."
            )

    return (
        "Unable to generate an answer."
    )


# -------------------------
# Test
# -------------------------

if __name__ == "__main__":

    pdf_path = (
        "data/attention-is-all-you-need.pdf"
    )

    chunks, index = prepare_pdf(
        pdf_path
    )

    print(
        "Number of chunks:",
        len(chunks)
    )

    print(
        "Vector store created:",
        index.ntotal
    )

    # Show first chunk as a test

    print("\nFirst chunk:")

    print(
        chunks[0]
    )