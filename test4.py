import kagglehub

# Download latest version
path = kagglehub.model_download("keras/gemma/keras/gemma_instruct_2b_en")

print("Path to model files:", path)