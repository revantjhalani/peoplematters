from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama
import redis
import fastapi

documents = SimpleDirectoryReader(
    input_files=["products.json"]
).load_data()

# bge embedding model
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

# ollama
Settings.llm = Ollama(model="mistral", request_timeout=30.0)

index = VectorStoreIndex.from_documents(
    documents,
)
preseed = "Limit yourself to only the JSON list provided and do not go out of scope and give me answer to the question: "
postseed = '? Use the JSON entries named features and description to formulate an answer, saGive me top 5 results, if there arent 5 then give only those that are relevant and do not show what doesnt exist in the provided JSON. Give the answer in the following JSON format where product-name is the product name, and reason is the reason that product is a good fit for the question asked [{"product-name" : "", "reason" : ""}, {"product-name" : "", "reason" : ""}, etc]. Only give responses with the product-name and the reason you formulate, use no other entry as a response. Give the answer in the provided JSON format only'

query_engine = index.as_query_engine()

response1 = str(query_engine.query(preseed + "What are the best mental wellness products" + postseed))
print(response1)