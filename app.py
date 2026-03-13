from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from regex import cache_all
import streamlit as st


st.title("Local RAG chatbot")
st.write("Ask questions from your PDF document.")

loader = PyPDFLoader("./youtube_guidelines.pdf")
documents= loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

vectorstore= Chroma.from_documents(documents = chunks,
                                   embedding = embeddings,
                                   persist_directory = "./chroma_db",
                                   collection_name="youtube_guidelines")

retriever = vectorstore.as_retriever(search_kwargs={'k':3})

llm= OllamaLLM(model="llama3")

prompt = ChatPromptTemplate.from_template(
    """
    You are a Helpful AI Assistant.
    
    Use the following retrieved context to answer the question.
    
    Context: {context}
    
    Question: {question}
    
    Answer consisely and accurately based on the provided context in no more than 3 sentences.
    """
    
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs,
     "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

question = st.text_input("Ask a question about the document: ")
if question:
    answer = rag_chain.invoke(question)
    st.write("Answer: ", answer)
