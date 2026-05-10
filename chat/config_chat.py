import os

# root path project
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
chat_dir = os.path.join(base_dir,"chat")
data_dir  = os.path.join(base_dir,"data")

model_path = os.path.join(chat_dir,"models","e5-base")

data_chat = os.path.join(data_dir,"raw","data_chat")


dir_vector = os.path.join(chat_dir,"chat_bot","faiss_db")




