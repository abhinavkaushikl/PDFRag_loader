# ✅ Install dependencies (already installed if you've followed our steps)
# pip install unstructured
# pip install langchain-core langchain-community langchain-text-splitters langchain-openai
# pip install sentence-transformers
# pip install nltk
# pip install tiktoken
# pip install pdfminer.six==20221105
# pip install pillow
# (Make sure Poppler is installed & in PATH)

import os
import re
import nltk
import tiktoken
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from sentence_transformers import SentenceTransformer

# ✅ Download nltk tokenizer once
nltk.download('punkt')

# ✅ Load PDF using Unstructured
pdf_path = "KomorowskiEDA2016.pdf"
loader = UnstructuredPDFLoader(pdf_path)
docs = loader.load()
document_text = " ".join([doc.page_content for doc in docs])

# ✅ Clean text function
def clean_text(raw_text):
    cleaned = re.sub(r'http\S+', '', raw_text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\[\d+\]', '', cleaned)
    return cleaned

cleaned_text = clean_text(document_text)

# ✅ Header-based splitting (structure-aware)
headers_to_split_on = [
    ("#", "Chapter"),
    ("##", "Section"),
    ("###", "Subsection"),
]

header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
header_chunks = header_splitter.split_text(cleaned_text)

# ✅ Extract actual text content from documents returned by splitter
header_texts = [doc.page_content for doc in header_chunks]

# ✅ Token counter helper
def token_count(text, model_name="cl100k_base"):
    encoding = tiktoken.get_encoding(model_name)
    return len(encoding.encode(text))

# ✅ RecursiveCharacterTextSplitter after header split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

base_chunks = []
for text in header_texts:
    base_chunks.extend(text_splitter.split_text(text))

# ✅ Use FREE sentence-transformers for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ Lightweight semantic filter (skip very small chunks)
semantic_chunks = [chunk for chunk in base_chunks if token_count(chunk) > 100]

# ✅ Metadata injection
final_chunks = []
for i, chunk in enumerate(semantic_chunks):
    metadata = {
        "document": "KomorowskiEDA2016",
        "chunk_index": i,
        "tokens": token_count(chunk)
    }
    final_chunks.append({
        "text": chunk,
        "metadata": metadata
    })

# ✅ Preview first few chunks
for fc in final_chunks[:3]:
    print("Chunk:\n", fc["text"])
    print("Metadata:\n", fc["metadata"])
    print("-" * 50)
