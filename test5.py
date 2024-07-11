import os 
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
hf_token = "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"
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import torch
# Define the repository ID for the Gemma 2b model
repo_id = "google/gemma-2b-it"

# Set up a Hugging Face Endpoint for Gemma 2b model
llm = HuggingFaceEndpoint(
    repo_id=repo_id,  temperature=0.1, max_new_tokens=50,
    model_kwargs={
        "token": hf_token,
        "torch_dtype": torch.float16,  # comment this line and uncomment below to use 4bit
        # "quantization_config": quantization_config
    },
)

question = "Who won the FIFA World Cup in the year 1994?"

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.invoke(question))

