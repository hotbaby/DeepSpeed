# encoding: utf8

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from pdfminer.high_level import extract_text



pdf_path = "/home/rd/Downloads/Megatron.pdf"
text = extract_text(pdf_path)


text_spliter = CharacterTextSplitter(separator="\n")
texts = text_spliter.split_text(text=text)
# print(texts)

prompt_template = """使用上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。

{context}

问题: {question}
中文答案:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff", prompt=PROMPT)

query = "What did the president say about Justice Breyer"
docs = []
chain({"input_documents": docs, "question": query}, return_only_outputs=True)