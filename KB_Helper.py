from langchain.llms import GooglePalm
from langchain.chains import RetrievalQA
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import UnstructuredExcelLoader   # for xls & xlsx files
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
llm = GooglePalm(google_api_key=os.environ['GOOGLE_API_KEY'], temperature=0.0)
instructor_embeddings = HuggingFaceInstructEmbeddings()

vector_db_path = "faiss_index"



def create_vectordb():
    xl_loader = UnstructuredExcelLoader("Hotel_KB_1.xlsx")
    data = xl_loader.load()
    # Create a FAISS instance for vector database from 'data'
    vector_db = FAISS.from_documents(documents = data,
                                     embedding = instructor_embeddings)
    vector_db.save_local(vector_db_path)


    
def quest_ans_chain():
    vector_db = FAISS.load_local(vector_db_path,instructor_embeddings)

    retriever = vector_db.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

if __name__ == "__main__":
    create_vectordb()
    chain = quest_ans_chain()
    print(chain("Can I get a cup of coffee?"))