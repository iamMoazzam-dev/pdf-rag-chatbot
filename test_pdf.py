from src.pdf_loader import load_pdf
from src.text_splitter import split_text


pdf_path = "data/attention-is-all-you-need.pdf"

# Step 1: Load PDF
text = load_pdf(pdf_path)

print("Extracted characters:", len(text))


# Step 2: Split text into chunks
chunks = split_text(text)

print("Number of chunks:", len(chunks))


# Step 3: Show first 3 chunks
for i, chunk in enumerate(chunks[:3], start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)