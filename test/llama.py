hf_token = "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"
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2b-it",
    token=hf_token,
)

stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

# generate_kwargs parameters are taken from https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
print("DONE1")
import torch
from llama_index.llms.huggingface import HuggingFaceLLM

# Optional quantization to 4bit
# import torch
# from transformers import BitsAndBytesConfig

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )

llm = HuggingFaceLLM(
    model_name="google/gemma-2b-it",
    context_window=8192,
    max_new_tokens=256,
    
    model_kwargs={
        "token": hf_token,
        "torch_dtype": torch.float16,  # comment this line and uncomment below to use 4bit
        # "quantization_config": quantization_config
    },
    generate_kwargs={
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
    },
    tokenizer_name="google/gemma-2b-it",
    tokenizer_kwargs={"token": hf_token, "max_length": 4096},
    stopping_ids=stopping_ids,
)
print("DONE2")
#response = llm.complete("hi how are you")
print("DONE3")
#print(response)

from llama_index.core.llms import ChatMessage

#messages = [
#    ChatMessage(role="system", content="You are CEO of MetaAI"),
#    ChatMessage(role="user", content="Introduce Llama3 to the world."),
#]
#response = llm.chat(messages)

print("DONE4")

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["product-list.json"]
).load_data()

print("DONE5")
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
print("DONE6")

from llama_index.core import Settings

# bge embedding model
Settings.embed_model = embed_model
Settings.chunk_size = 512
# Llama-3-8B-Instruct model
Settings.llm = llm


index = VectorStoreIndex.from_documents(
    documents,
)
print("DONE7")


query_engine = index.as_query_engine(similarity_top_k=3)

import os 
os.system("date")
response = query_engine.query("what are the top hrms products")
os.system("date")
print("DONE8")
print(response)


