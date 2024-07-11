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
Settings.llm = Ollama(model="phi3:mini", request_timeout=30.0)

index = VectorStoreIndex.from_documents(
    documents,
)
preseed = "Limit yourself to only the JSON provided and do not go out of scope and give me answer to the question: "
postseed = '? Give me top 5 results, if there arent 5 then give only those that are relevant and do not show what doesnt exist in the provided JSON. Give the answer in the following JSON format where product is the product name, and reason is the reason that product is a good fit for the question asked [{"product" : "", "reason" : ""}, {"product" : "", "reason" : ""}, etc]. Give the answer in the provided JSON format only'

query_engine = index.as_query_engine()
#response = query_engine.query(preseed +  "Best hrms solutions for mid sized companies" + postseed)
#print(response)

# response = query_engine.query("Best software’s for employee productivity")
# print(response)


# response = query_engine.query("Best software’s for employee welfare")
# print(response)

from typing import Union
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import redis
import requests 
import json
#r = redis.Redis(host='localhost', password="g4QLiT2ELqs3986P2e8xQYGRZlbLKjYMj5qYP7r1dq+TtjodCyV+r1YZYZE8TWx0hmDU7+qROeLsqSNq", port=6379, db=1, decode_responses=True)
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/peoplematters")
def mlresponse(query: Union[str, None] = None):
    print("recieved query: ", query)
    response1 = str(query_engine.query(preseed +  str(query) + postseed))
    print(response1)
    #response1 = json.loads(response1)["response"]
    return response1


