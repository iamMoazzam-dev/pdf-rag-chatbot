import gradio as gr

from src.rag import (
    prepare_pdf,
    load_saved_rag,
    retrieve_chunks,
    generate_answer
)


# -------------------------
# Global RAG variables
# -------------------------

chunks = None
index = None


# -------------------------
# Load saved RAG data
# -------------------------

print("Checking for saved RAG data...")

saved_chunks, saved_index = load_saved_rag()

if saved_chunks is not None and saved_index is not None:

    chunks = saved_chunks
    index = saved_index

    print("Saved RAG data loaded!")
    print(
        "Number of chunks:",
        len(chunks)
    )

else:

    print("No saved RAG data found.")


# -------------------------
# Process uploaded PDF
# -------------------------

def process_pdf(file):

    global chunks, index

    if file is None:

        return "Please upload a PDF first."

    try:

        print("\nProcessing PDF...")

        chunks, index = prepare_pdf(
            file
        )

        print(
            "PDF processed successfully!"
        )

        print(
            "Number of chunks:",
            len(chunks)
        )

        return (
            f"PDF processed successfully! "
            f"{len(chunks)} chunks created."
        )

    except Exception as error:

        print(
            "\nPDF processing failed."
        )

        print(error)

        chunks = None
        index = None

        return (
            f"Could not process PDF: {error}"
        )


# -------------------------
# Chat with PDF
# -------------------------

def chat_with_pdf(
    message,
    history
):

    global chunks, index

    if not message.strip():

        return history, ""


    # -------------------------
    # Check PDF
    # -------------------------

    if chunks is None or index is None:

        history.append({
            "role": "assistant",
            "content": (
                "Please upload and process "
                "a PDF first."
            )
        })

        return history, ""


    try:

        # -------------------------
        # Retrieve relevant chunks
        # -------------------------

        retrieved_chunks = retrieve_chunks(
            message,
            index,
            chunks
        )


        # -------------------------
        # Generate answer
        # -------------------------

        answer = generate_answer(
            message,
            retrieved_chunks
        )


        # -------------------------
        # Get source pages
        # -------------------------

        source_pages = []

        for chunk in retrieved_chunks:

            page_number = chunk["page"]

            if page_number not in source_pages:

                source_pages.append(
                    page_number
                )


        # -------------------------
        # Create source section
        # -------------------------

        if source_pages:

            sources = ", ".join(
                [
                    f"Page {page}"
                    for page in source_pages
                ]
            )

        else:

            sources = "No source pages found."


        # -------------------------
        # Final answer
        # -------------------------

        final_answer = (
            f"{answer}\n\n"
            f"**Sources:** {sources}"
        )


        # -------------------------
        # Add user message
        # -------------------------

        history.append({
            "role": "user",
            "content": message
        })


        # -------------------------
        # Add AI response
        # -------------------------

        history.append({
            "role": "assistant",
            "content": final_answer
        })


        return history, ""


    except Exception as error:

        print(
            "\nChat error:"
        )

        print(error)

        history.append({
            "role": "assistant",
            "content": (
                "An error occurred while "
                "processing your question."
            )
        })

        return history, ""


# -------------------------
# Clear chat
# -------------------------

def clear_chat():

    return []


# -------------------------
# Gradio UI
# -------------------------

with gr.Blocks(
    title="PDF RAG Chatbot"
) as demo:


    # -------------------------
    # Title
    # -------------------------

    gr.Markdown(
        """
        # PDF RAG Chatbot

        Upload a PDF and ask questions
        about its content.
        """
    )


    # -------------------------
    # PDF Upload
    # -------------------------

    pdf_file = gr.File(
        label="Upload PDF",
        file_types=[".pdf"],
        type="filepath"
    )


    process_button = gr.Button(
        "Process PDF"
    )


    status = gr.Textbox(
        label="Status",
        interactive=False
    )


    # -------------------------
    # Chatbot
    # -------------------------

    chatbot = gr.Chatbot(
        label="Conversation",
        height=500
    )


    # -------------------------
    # Question input
    # -------------------------

    message = gr.Textbox(
        label="Your Question",
        placeholder=(
            "Ask something about the PDF..."
        ),
        lines=2
    )


    ask_button = gr.Button(
        "Ask"
    )


    clear_button = gr.Button(
        "Clear"
    )


    # -------------------------
    # Process PDF button
    # -------------------------

    process_button.click(
        fn=process_pdf,
        inputs=pdf_file,
        outputs=status
    )


    # -------------------------
    # Ask button
    # -------------------------

    ask_button.click(
        fn=chat_with_pdf,
        inputs=[
            message,
            chatbot
        ],
        outputs=[
            chatbot,
            message
        ]
    )


    # -------------------------
    # Enter key
    # -------------------------

    message.submit(
        fn=chat_with_pdf,
        inputs=[
            message,
            chatbot
        ],
        outputs=[
            chatbot,
            message
        ]
    )


    # -------------------------
    # Clear button
    # -------------------------

    clear_button.click(
        fn=clear_chat,
        inputs=None,
        outputs=chatbot
    )


# -------------------------
# Launch application
# -------------------------

demo.launch()