from langchain_community.llms.ollama import Ollama
import time

llm = Ollama(model="llama3")

def run_chain(chain, prompt): 
    tokens = []
    start = time.time()
    for chunk in chain.stream(prompt): 
        print(chunk, end="", flush=True)
        tokens.append(chunk)
    end = time.time()
    print(f"\n\n {len(tokens)/(end-start)}")

run_chain(llm,"shortly explain Nvidia Cuda using emoji's")
