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

# kiểm tra xem có dữ liệu đã đc lưu trong máy chưa
if not os.path.exists(db_faiss):
    raise ValueError(" Chưa có FAISS DB, hãy chạy vector_pipeline.py trước!")

# khởi tạo model embeđing
embeddings = HuggingFaceEmbeddings(
    model_name=config_chat.model_path,
    encode_kwargs={"normalize_embeddings": True}
)

# load dữ liệu vào vectorstore
vectorstore = FAISS.load_local(
    db_faiss,
    embeddings,
    allow_dangerous_deserialization=True
)

# ===== RETRIEVER =====
# retriever = vectorstore.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "k": 2,
#         "fetch_k": 4,
#         "lambda_mult": 0.7
#     }
# )

# Khởi tạo retriever để tìm các chunk liên quan tới câu hỏi
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

# Format các documents thành context đưa vào prompt
def format_docs(docs: list[Document]):
    return "\n\n".join(
        f"[Tài liệu {i+1}]:\n{doc.page_content[:700]}"
        for i, doc in enumerate(docs)
    )

# tạo promt để chát bot trả lời theo yêu cầu của mình
template = """
Bạn là trợ lý y khoa chuyên về bệnh võng mạc do tiểu đường.

QUY TẮC:
- Chỉ trả lời dựa trên CONTEXT.
- Không suy đoán.
- Không bịa thông tin.
- Nếu không có dữ liệu → "Tôi không biết".
- Trả lời bằng tiếng Việt dễ hiểu cho bệnh nhân.
- KHÔNG đưa nguồn hoặc link nếu người dùng không yêu cầu.
- Không đưa chẩn đoán cuối cùng, khuyên gặp bác sĩ khi cần.

CONTEXT:
{context}

CÂU HỎI:
{question}

TRẢ LỜI:
"""
prompt = ChatPromptTemplate.from_template(template)


# khởi tạo model
llm = ChatOllama(
    model="qwen2:7b",
    temperature=0,
    num_ctx=512,
    num_predict=128
)

# tổng kết lại các bước thành pipline hoàn chỉnh
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Hàm gọi RAG pipeline để lấy câu trả lời
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