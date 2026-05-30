# ingestion.py
# This module takes care of reading documents (PDFs and text files)
# and breaking them into smaller chunks that we can embed and search later.
#
# We use PyMuPDF for PDFs because it's fast and doesn't need Java or Poppler.
# For chunking, LangChain's RecursiveCharacterTextSplitter does a nice job
# of splitting on paragraph/sentence boundaries instead of cutting mid-word.

import os

import fitz  # PyMuPDF
import docx
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


# -- Reading files -----------------------------------------------------------

def extract_text_from_pdf(file_path: str) -> str:
    # open the PDF and grab text from every page
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def extract_text_from_txt(file_path: str) -> str:
    # just read the whole file as a string
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text_from_docx(file_path: str) -> str:
    # extract all paragraphs and join them with newlines
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_html(file_path: str) -> str:
    # use BeautifulSoup to strip HTML tags and scripts, leaving just the text
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
        return soup.get_text(separator="\n", strip=True)


def load_document(file_path: str) -> str:
    # pick the right reader based on the file extension
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".txt", ".md"]:
        return extract_text_from_txt(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".html":
        return extract_text_from_html(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .txt, .docx, .md, .html")


# -- Chunking ----------------------------------------------------------------

def chunk_text(text: str) -> list[str]:
    # split the text into overlapping pieces
    # the splitter tries paragraph breaks first, then newlines, then sentences,
    # so chunks end up on natural boundaries most of the time
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


# -- Main entry point --------------------------------------------------------

def ingest_document(file_path: str) -> list[str]:
    # full pipeline: read the file, then chop it into chunks
    text = load_document(file_path)
    chunks = chunk_text(text)
    return chunks
