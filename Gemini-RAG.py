import getpass
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from markitdown import MarkItDown
from langchain.schema import Document

md = MarkItDown()
result = md.convert("tested.csv")

docs = [Document(page_content=result.text_content)]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
split_docs = text_splitter.split_documents(docs)

google_api_key = ""

embeddings = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model="models/embedding-001")

vectorstore = Chroma.from_documents(documents=split_docs, embedding=GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model="models/embedding-001"))

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

llm = ChatGoogleGenerativeAI(google_api_key=google_api_key, model="gemini-1.5-pro", temperature=0.3, max_tokens=500)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "the answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "Tell me about this dataset"})
print(response["answer"])
