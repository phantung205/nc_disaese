from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from chat import config_chat
import os

db_faiss = config_chat.dir_vector

# ===== CHECK FILE =====
if not os.path.exists(db_faiss):
    raise ValueError(" Chưa có FAISS DB, hãy chạy vector_pipeline.py trước!")

print("Files in DB:", os.listdir(db_faiss))

# ===== EMBEDDING =====
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ===== LOAD DB =====
vectorstore = FAISS.load_local(
    db_faiss,
    embeddings,
    allow_dangerous_deserialization=True
)

# ===== RETRIEVER =====
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

# ===== FORMAT DOCS =====
def format_docs(docs: list[Document]):
    return "\n\n".join(doc.page_content for doc in docs)

# ===== PROMPT =====
template = """
You are a medical assistant specializing in diabetic retinopathy.

Answer ONLY based on the context below.
If the answer is not in the context, say: "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

# ===== LLM =====
llm = ChatOllama(
    model="llama3",
    temperature=0
)

# ===== RAG =====
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

def get_answer(question: str) -> str:
    return rag_chain.invoke(question)

# ===== CHAT =====
if __name__ == "__main__":
    while True:
        question = input("\n Question: ")

        if question.lower() in ["exit", "quit"]:
            break

        answer = get_answer(question)

        print("\n Answer:\n", answer)