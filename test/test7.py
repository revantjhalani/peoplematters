from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama
import redis
import fastapi

documents = SimpleDirectoryReader(
    input_files=["/workspace/peoplematters/test/products.json"]
).load_data()

# bge embedding model
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

# ollama
Settings.llm = Ollama(model="mistral", request_timeout=30.0)

index1 = VectorStoreIndex.from_documents(
    documents,
)
# index2 = VectorStoreIndex.from_documents(
#     documents,
# )
print("Loaded")
chat_engine1 = index1.as_chat_engine()
# chat_engine2 = index2.as_chat_engine()
# response = chat_engine1.chat("tell me the best time management products")
# print(response)
# response = chat_engine1.chat("no Xemplo also does time management, provide me the previous produts with Xemplo")
# print(response)

preseed = "Limit yourself to only the JSON provided and do not go out of scope and give me answer to the question in JSON format provided at the end of this question: "
postseed = '? Give me top 5 results, if there arent 5 then give only those that are relevant and do not show what doesnt exist in the provided JSON. Give the answer in the following JSON format where product is the product name(the name field in the json), and reason is the reason that product is a good fit for the question asked [{"product" : "", "reason" : ""}, {"product" : "", "reason" : ""}, etc]. Give the answer in the provided JSON format only'

seed = 'The provided JSON is a list of products, the products being the name field in the json, only provide the name of the product, not the feature. Limit yourself to only the JSON provided, Give the top 5 results, if there arent 5 then give only those that are relevant and do not show what doesnt exist in the provided JSON. Give the answer in the following JSON format where product is the product name(the name field in the json, not features), and reason is the reason that product is a good fit for the question asked [{"product" : "", "reason" : ""}, {"product" : "", "reason" : ""}, ... ]. Give the answer in the provided JSON format only The question is: '
question = "what are the best mental wellness products"

# while True:
print("Question chat1 : ")
response = chat_engine1.chat(seed + question)
print("Answer chat1 : ")
print(response)
print("Follow up : ")
response = chat_engine1.chat(input())

print(response)

    # print("Question chat2 : ")
    # response = chat_engine2.chat(input())
    # print("Answer chat2 : ")
    # print(response)
    