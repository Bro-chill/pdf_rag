import streamlit as st
import os
import logging
import ollama

from langchain_community.document_loaders import UnstructuredPDFLoader

DOC_PATH = ["./data/Act_505.pdf", "./data/Act_559.pdf", "./data/Federal_Constitution.pdf"]

# Loggin configuration
logging.basicConfig(level=logging.INFO)

def ingest_pdf(doc_path):
    if doc_path:
        documents = []
        for pdf in doc_path:
            loader = UnstructuredPDFLoader(pdf)
            docs = loader.load()
            documents.extend(docs)
        logging.info("Loading pdf files complete")
        return documents
    else:
        logging.error("Loading pdf error")
        return []

from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents=documents)
    logging.info("Chunking complete")
    return chunks

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-law-rag2"
PERSIST_DIRECTORY = "./chroma_db"

@st.cache_resource
def load_vector_db():
    """Load vector to database"""
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
    data = ingest_pdf(DOC_PATH)
    chunks = split_documents(data)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name=VECTOR_STORE_NAME,
        persist_directory=PERSIST_DIRECTORY,
    )
    vector_db.persist()
    logging.info("Vector database created and persisted")
    return vector_db

from langchain.prompts  import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

def create_retriever(vector_db, llm):
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    """Retrieving"""
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        prompt=QUERY_PROMPT
    )
    logging.info("Retrieving Complete")
    return retriever

def create_chain(retriever, llm):
    template = """Answer the question based ONLY on the following context:
                {context}
                Question: {question}
                """
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever,
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logging.info("Chained created")
    return chain

LLM_MODEL = "llama3.2"

def main():
    st.title("Document Assistant")

    # Input
    user_input = st.text_input("Enter your question", "")

    if user_input:
        with st.spinner("Generating response.."):
            try:
                # Initialize llm model
                llm = ChatOllama(model=LLM_MODEL)

                # Load vector database
                vector_db = load_vector_db()
                if vector_db is None:
                    st.error("Failed to load/create vector database")
                    return
                
                # Create retriever
                retriever = create_retriever(vector_db, llm)

                # Create chain
                chain = create_chain(retriever, llm)

                # Get response
                response = chain.invoke(input=user_input)

                st.markdown("**Assistant:**")
                st.write(response)
            except Exception as e:
                st.error(f"Error occured: {str(e)}")
    else:
        st.error("Please enter your question")

if __name__ ==  "__main__":
    main()