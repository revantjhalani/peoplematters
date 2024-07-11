from llama_index.llms.ollama import Ollama
gemma_2b = Ollama(model="gemma:2b", request_timeout=30.0)

resp = gemma_2b.complete("Who is Paul Graham?")
print(resp)
