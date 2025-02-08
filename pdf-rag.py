from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

doc_path = ["./data/Act_505.pdf", "./data/Act_559.pdf", "./data/Federal_Constitution.pdf"]
model = "llama3.2"

# Upload PDF file locally
if doc_path:
    documents = []
    for pdf in doc_path:
        loader = UnstructuredPDFLoader(pdf)
        docs = loader.load()
        documents.extend(docs)
    print("Loading complete")
else:
    print("Loading error")

# Preview loaded data
# content = documents[2].page_content
# print(content[:100])

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Split and Chunk text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks = text_splitter.split_documents(documents)
print("Chunk and Split Text Complete")
# print(f"Number of chunks: {len(chunks)}")
# print(f"Example chunk: {chunks[0]}")

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import ollama

# Add chunks into vector database
ollama.pull("nomic-embed-text")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="simple-law-rag",
)
print("Adding to vector database complete")

from langchain.prompts  import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Set LLM model to use
llm = ChatOllama(model=model)

# Augmenting prompt
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

# Retrieving from vector database
retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),
    llm,
    prompt=QUERY_PROMPT
)

# Generating prompt
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

res = chain.invoke(input=("What is syariah criminal offense?"))
print(res)