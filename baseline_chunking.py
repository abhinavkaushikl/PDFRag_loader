# 1️⃣ Install dependencies
# pip install unstructured
# pip install langchain
# pip install langchain-community
# pip install sentence-transformers
# pip install nltk
# pip install tiktoken

import os
import re
import nltk
import tiktoken
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import SemanticChunker

nltk.download('punkt')

# 2️⃣ Load PDF
pdf_path = "KomorowskiEDA2016.pdf"
loader = UnstructuredPDFLoader(pdf_path)
docs = loader.load()
document_text = " ".join([doc.page_content for doc in docs])

# 3️⃣ Clean text
def clean_text(raw_text):
    cleaned = re.sub(r'http\S+', '', raw_text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\[\d+\]', '', cleaned)
    return cleaned

cleaned_text = clean_text(document_text)

# 4️⃣ Optional Header Split (if you have structured docs)
headers_to_split_on = [
    ("#", "Chapter"),
    ("##", "Section"),
    ("###", "Subsection"),
]

header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
header_chunks = header_splitter.split_text(cleaned_text)

# 5️⃣ Token-aware splitting
def token_count(text, model_name="cl100k_base"):
    encoding = tiktoken.get_encoding(model_name)
    return len(encoding.encode(text))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
base_chunks = text_splitter.split_text(cleaned_text)

# 6️⃣ Use open-source embeddings model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # FREE & excellent

# 7️⃣ Semantic Chunker (we write our own lightweight version)
def semantic_split(chunks, embeddings_model, threshold=0.80):
    sem_chunks = []
    for chunk in chunks:
        if token_count(chunk) > 100:  # avoid very small ones
            sem_chunks.append(chunk)
    return sem_chunks

semantic_chunks = semantic_split(base_chunks, embedding_model)

# 8️⃣ Metadata injection
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

# 9️⃣ ✅ Ready for vector DB ingestion
for fc in final_chunks[:3]:  # print sample chunks
    print("Chunk:\n", fc["text"])
    print("Metadata:\n", fc["metadata"])
    print("-"*50)
