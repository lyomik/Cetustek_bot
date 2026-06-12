# build_sop_db.py
import pdfplumber
import os
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


SOP_DIR = "./PDF_sop"       # 放所有 SOP PDF 的資料夾
CHROMA_DIR = "./chroma_sop"   # 向量庫存放位置

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"}
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

all_docs = []

for filename in os.listdir(SOP_DIR):
    if not filename.endswith(".pdf"):
        continue
    filepath = os.path.join(SOP_DIR, filename)
    print(f"處理中：{filename}")
    
    with pdfplumber.open(filepath) as pdf:
        text = "\n".join(
            p.extract_text() for p in pdf.pages if p.extract_text()
        )
    
    chunks = splitter.create_documents(
        [text],
        metadatas=[{"source": filename}]  # 記錄來源檔名
    )
    all_docs.extend(chunks)
    print(f"  → {len(chunks)} 個段落")

# 建立並持久化向量庫
vector_db = Chroma.from_documents(
    all_docs,
    embeddings,
    persist_directory=CHROMA_DIR
)

print(f"\n完成！共 {vector_db._collection.count()} 筆段落存入 {CHROMA_DIR}")