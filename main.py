""""""

import ollama
import os
import fitz
from dbs.redis_stack import RedisStack

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


# extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


def process_pdfs(data_dir: str, store):
    """ """

    for file_name in os.listdir(data_dir):  # iterate over all pdf files
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(
                pdf_path
            )  # use pdf library to extract text by page from pdf
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(
                    text
                )  # page by page chunking text, extract certain number of characters to be vectorized
                # print(f"  Chunks: {chunks}")
                for chunk_index, chunk in enumerate(chunks):
                    # embedding = calculate_embedding(chunk)
                    embedding = get_embedding(
                        chunk
                    )  # calculate embedding for each chunk and then store
                    store(
                        file=file_name,
                        page=str(page_num),
                        # chunk=str(chunk_index),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def main():
    redis_db = RedisStack(VECTOR_DIM, INDEX_NAME, DOC_PREFIX, DISTANCE_METRIC)
    process_pdfs("", redis_db.store)


if __name__ == "__main__":
    main()
