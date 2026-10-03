from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(pages):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []

    for page in pages:

        page_text = page["text"]

        page_number = page["page"]

        page_chunks = splitter.split_text(
            page_text
        )

        for chunk in page_chunks:

            chunks.append({
                "page": page_number,
                "text": chunk
            })

    return chunks