# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
gemma_lm = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    max_length=1000
    
)

# input_text = "Write me a poem about Machine Learning."
# input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# outputs = model.generate(**input_ids)
# print(tokenizer.decode(outputs[0]))

from typing import Any

from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback

class Gemma(CustomLLM):
    num_output: int = 512
    model_name: str = "Gemma"
    model: Any = None
    
    def __init__(self, model, num_output):
        super(Gemma, self).__init__()
        self.model = model
        self.num_output = num_output

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text=self.model.generate(prompt, max_length=self.num_output))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        response = ""
        for token in self.model.generate(prompt, max_length=self.num_output):
            response += token
            yield CompletionResponse(text=response, delta=token)
            


from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = Gemma(gemma_lm, 512)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")



from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["product-list.json"]
).load_data()



# print("DONE5")
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# print("DONE6")

# from llama_index.core import Settings

# # bge embedding model
# Settings.embed_model = embed_model
# Settings.chunk_size = 512
# # Llama-3-8B-Instruct model



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
