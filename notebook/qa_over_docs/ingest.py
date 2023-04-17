# encoding: utf8

import faiss
import pickle
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from pdfminer.high_level import extract_text


ps = list(Path("/data/datasets/papers").glob("**/*.pdf"))

data = []
sources = []

for p in ps[:3]:
    # with open(p) as f:
    data.append(extract_text(p))
    sources.append(p)

    print(f"parse pdf {p}")
    

text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))


store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)