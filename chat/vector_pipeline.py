from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from chat import config_chat
import os


loader = DirectoryLoader(
    path=config_chat.data_chat,
    glob="**/*.pdf",
    loader_cls=UnstructuredFileLoader,
)

docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
)

splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(
    documents=splits,
    embedding=embeddings,
    distance_strategy=DistanceStrategy.COSINE
)

os.makedirs(config_chat.dir_vector, exist_ok=True)

vectorstore.save_local(config_chat.dir_vector)
