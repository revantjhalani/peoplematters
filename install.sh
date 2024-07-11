
apt update 
apt install -y python3 python3-pip python3-dev python3-setuptools build-essential pciutils lshw tmux
curl -fsSL https://ollama.com/install.sh | sh
pip3 install redis wget llama-index llama-index-core llama-index-readers-file llama-index-llms-ollama llama-index-embeddings-huggingface fastapi
