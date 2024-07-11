from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama
from llama_index.core.indices.struct_store import JSONQueryEngine
import json

documents = SimpleDirectoryReader(
    input_files=["products.json"]
).load_data()

# bge embedding model
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

# ollama
Settings.llm = Ollama(model="mistral", request_timeout=30.0)

products = open('products.json')
schema = open("schema.json")


nl_query_engine = JSONQueryEngine(
    json_value=products,
    json_schema=schema,
    llm=Settings.llm,
)

nl_response = nl_query_engine.query(
    'Limit yourself to only the JSON provided and do not go out of scope and give me answer to the question: what are the best time tracking products? Give me top 5 results, if there arent 5 then give only those that are relevant and do not show what doesnt exist in the provided JSON. Give the answer in the following JSON format where product is the product name, and reason is the reason that product is a good fit for the question asked [{"product" : "", "reason" : ""}, {"product" : "", "reason" : ""}, etc]. Give the answer in the provided JSON format only ',
)

print(nl_response)
# raw_query_engine = JSONQueryEngine(
#     json_value=json_value,
#     json_schema=json_schema,
#     llm=llm,
#     synthesize_response=False,
# )