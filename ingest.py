""""""

import ollama
import os
import fitz
from dbs.redis_stack import RedisStack

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


def extract_text_from_pdf(pdf_path: str):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    """ """
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def process_docs(data_dir: str, store):
    """
    Process all the documents in a given directory
    :data_dir str: the directory with all the data files
    :store lambda: a VDatabase store method to store the vectors
    """

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)
                    store(
                        file=file_name,
                        page=str(page_num),
                        # chunk=str(chunk_index),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")


def main():
    redis_db = RedisStack(VECTOR_DIM, INDEX_NAME, DOC_PREFIX, DISTANCE_METRIC)
    redis_db.clear()
    process_docs("./class_notes", redis_db.store)


if __name__ == "__main__":
    main()
