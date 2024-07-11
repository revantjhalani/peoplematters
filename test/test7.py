from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama
import redis
import fastapi

documents = SimpleDirectoryReader(
    input_files=["/root/workspace/peoplematters/test/products.json"]
).load_data()

# bge embedding model
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

# ollama
Settings.llm = Ollama(model="mistral", request_timeout=30.0)

index1 = VectorStoreIndex.from_documents(
    documents,
)
index2 = VectorStoreIndex.from_documents(
    documents,
)
print("Loaded")
chat_engine1 = index1.as_chat_engine()
chat_engine2 = index2.as_chat_engine()
# response = chat_engine1.chat("tell me the best time management products")
# print(response)
# response = chat_engine1.chat("no Xemplo also does time management, provide me the previous produts with Xemplo")
# print(response)
while True:
    print("Question chat1 : ")
    response = chat_engine1.chat(input())
    print("Answer chat1 : ")
    print(response)
    print("Question chat2 : ")
    response = chat_engine2.chat(input())
    print("Answer chat2 : ")
    print(response)
    