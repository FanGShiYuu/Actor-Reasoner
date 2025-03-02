# Interact, Instruct to Improve: A LLM-Driven Parallel Actor-Reasoner Framework for Enhancing Autonomous Vehicle Interactions
[Shiyu Fang](https://fangshiyuu.github.io/)

## Getting started 🚀

#### 1. Install Ollama
Before you start, make sure to have `ollama` installed and a working environment. If you don’t have `ollama` installed, check here [Ollama](https://ollama.com/):
After installing Ollama, run the following command to install llama3:
```shell
ollama run llama3
```
You can choose any other LLM model provided by Ollama—just replace "llama3" with the model you want to use and change llm_agent.py line10 accordingly. Check [here](https://ollama.com/search) for more details!

#### 2. Install the dependent package
```shell
pip install -r requirements.txt
```

#### 3. Start training and testing
```shell
python train.py
python test.py
python test_with_listener.py
```
