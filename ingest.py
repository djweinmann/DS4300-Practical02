""""""

import os
import fitz
from utils.parse_args import get_database, get_ingestion
from utils.timer import timer

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


def process_docs(data_dir: str, store, chunk_size, overlap):
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
                chunks = split_text_into_chunks(text, chunk_size, overlap)
                for chunk_index, chunk in enumerate(chunks):
                    store(
                        file=file_name,
                        page=str(page_num),
                        # chunk=str(chunk_index),
                        chunk=str(chunk),
                    )
            print(f" -----> Processed {file_name}")

@timer
def main():
    db = get_database()
    db.clear()

    chunk_size, overlap = get_ingestion()

    process_docs("./class_notes/", db.store, chunk_size, overlap)


if __name__ == "__main__":
    main()
