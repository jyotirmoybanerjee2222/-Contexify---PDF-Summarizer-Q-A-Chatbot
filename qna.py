# qna.py
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain as langchain_load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables and configure Gemini
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory="chroma_db")
    vectorstore.persist()

def build_qa_chain():
    template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, just say 
    "Answer is not available in the context", don't make up an answer.

    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    return langchain_load_qa_chain(model, chain_type="stuff", prompt=prompt)

def ask_question(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    docs = vectorstore.similarity_search(question)
    chain = build_qa_chain()
    return chain({"input_documents": docs, "question": question}, return_only_outputs=True)["output_text"]
