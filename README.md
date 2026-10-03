# PDF RAG Chatbot

A beginner-friendly Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF and ask questions about its content.

The application extracts text from the PDF, divides the text into smaller chunks, converts those chunks into embeddings, stores them in a FAISS vector database, retrieves the most relevant chunks for a user's question, and uses Google's Gemini API to generate an answer based on the retrieved information.

## How It Works

```text
PDF
 ↓
Extract Text
 ↓
Split Text into Chunks
 ↓
Create Embeddings
 ↓
Store Embeddings in FAISS
 ↓
User Asks a Question
 ↓
Create Question Embedding
 ↓
Search Relevant Chunks
 ↓
Send Chunks + Question to Gemini
 ↓
Generate Answer
 ↓
Show Answer + Source Pages
```

## Features

* Upload a PDF file
* Extract text from PDF pages
* Split documents into smaller chunks
* Generate text embeddings
* Store embeddings using FAISS
* Perform semantic similarity search
* Retrieve relevant PDF content
* Generate answers using Gemini
* Display relevant PDF page numbers
* Handle PDFs with no extractable text
* Simple Gradio web interface
* Local RAG data storage for faster application startup

## Technologies Used

* Python
* Gradio
* PyPDF
* LangChain Text Splitters
* Sentence Transformers
* FAISS
* Google Gemini API
* python-dotenv

## Project Structure

```text
pdf-rag-chatbot/
│
├── data/
│   └── .gitkeep
│
├── src/
│   ├── embeddings.py
│   ├── pdf_loader.py
│   ├── rag.py
│   ├── text_splitter.py
│   └── vector_store.py
│
├── .gitignore
├── main.py
├── README.md
├── requirements.txt
│
├── test_embeddings.py
├── test_pdf.py
└── test_vector_store.py
```

## RAG Pipeline

### 1. PDF Loading

`pdf_loader.py` uses PyPDF to extract text from each PDF page.

Each page is stored together with its page number so that the application can later show the source page.

### 2. Text Splitting

The extracted text is divided into smaller chunks using a recursive text splitter.

Current configuration:

```text
Chunk size: 1000 characters
Chunk overlap: 200 characters
```

The overlap helps preserve context between neighboring chunks.

### 3. Embeddings

The project uses:

```text
all-MiniLM-L6-v2
```

from Sentence Transformers.

The embedding model converts text into numerical vectors that represent the semantic meaning of the text.

### 4. Vector Database

FAISS is used to store the generated vectors.

When a user asks a question, the question is also converted into an embedding.

FAISS then searches for the most similar document chunks.

### 5. Generation

The retrieved chunks are provided to Google's Gemini API together with the user's question.

Gemini generates the final answer using the retrieved PDF information as context.

### 6. Source Pages

The application keeps the original page number with every chunk.

This allows the interface to show the PDF pages that were retrieved for the answer.

## Installation

Clone the repository and enter the project directory:

```bash
git clone <your-repository-url>
cd pdf-rag-chatbot
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

### macOS / Linux

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_gemini_api_key
```

The `.env` file is ignored by Git and should never be uploaded to GitHub.

## Run the Application

Make sure the virtual environment is activated:

```bash
source .venv/bin/activate
```

Then run:

```bash
python main.py
```

The Gradio application will start locally.

Open the local URL shown in the terminal.

## Using the Application

1. Open the application.
2. Upload a PDF.
3. Click **Process PDF**.
4. Wait for the PDF to be processed.
5. Enter a question about the PDF.
6. Click **Ask**.
7. The application retrieves relevant sections and generates an answer.
8. Relevant PDF page numbers are displayed below the answer.

## Example

For a research paper about Transformer models, a user could ask:

```text
What is the main idea behind the Transformer architecture?
```

The application retrieves relevant sections from the uploaded PDF and sends them to Gemini as context.

## Error Handling

The application handles several common situations, including:

* No PDF uploaded
* PDF with no extractable text
* Missing Gemini API key
* Gemini quota errors
* Temporary Gemini service errors
* Empty questions

## Important Notes

This project uses a local FAISS index and local chunk storage.

The generated files are intentionally excluded from Git:

```text
data/*.pdf
data/faiss.index
data/chunks.pkl
```

This prevents large PDF files and generated vector-store files from being uploaded to the Git repository.

## Limitations

* The application works best with text-based PDFs.
* Scanned/image-only PDFs may not contain extractable text.
* The quality of answers depends on the quality of the retrieved chunks.
* Gemini API availability and quota can affect answer generation.
* The current application retrieves a small number of relevant chunks for each question.
* Local FAISS data must be recreated when processing a new PDF.

## What I Learned

This project was built to understand the fundamentals of Retrieval-Augmented Generation.

Key concepts practiced:

* PDF text extraction
* Text chunking
* Embeddings
* Semantic similarity
* Vector databases
* FAISS
* Retrieval
* Prompt construction
* LLM API integration
* Gemini API
* Environment variables
* `.env` files
* Python virtual environments
* Modular Python project structure
* Gradio interfaces
* Git and GitHub

## Future Improvements

Possible improvements include:

* Support for multiple PDFs
* Conversation memory
* Better source citations
* Displaying retrieved text snippets
* Improved document chunking
* Streaming responses
* Authentication
* Docker deployment
* Cloud deployment
* More robust handling of scanned PDFs
* Support for additional LLM providers

## Author

Moazzam

This project is part of my hands-on journey into Generative AI and RAG application development.
