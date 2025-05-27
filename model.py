from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="deepseek-r1:1.5b")
try:
    response = llm.invoke("Hello!")
    print("LLM test response:", response)
except Exception as e:
    print(f"Ollama test failed: {e}")