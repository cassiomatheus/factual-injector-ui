from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def get_qwen_model(temperature=0.2):
    """
    Inicializa o Qwen 3.5 9B local via Ollama. 
    Temperatura baixa para maior precisão e determinismo na injeção.
    """
    llm = Ollama(
        model="qwen3.5:9b",
        temperature=temperature,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
    return llm