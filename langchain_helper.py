import os
import warnings
from dotenv import load_dotenv

# Modern v1.x imports
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# Suppress non-critical warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

load_dotenv(override=True)

# 1. Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="models/gemini-3.6-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# 2. Fast CPU Embeddings (Avoids 'INSTRUCTOR' object AttributeError)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb_file_path = "faiss_fitness_index"

def create_vector_db():
    """Builds and saves local FAISS vector index from CSV data."""
    loader = CSVLoader(file_path='fitness_data.csv', source_column="name")
    data = loader.load()

    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(vectordb_file_path)
    print("FAISS Index created successfully!")

def get_qa_chain():
    """Loads FAISS index and returns RetrievalQA chain."""
    vectordb = FAISS.load_local(
        vectordb_file_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    # Top-k retriever ensures documents are always fetched reliably
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
In the answer try to provide as much text as possible from the "details" and "information" sections in the source document context without making significant changes.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

CONTEXT: {context}

QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    response = chain.invoke({"query": "What are the macros for Chicken Breast?"})
    print("\nResult:")
    print(response["result"])