from utils.vprint import vprint
import ollama


class Generator:
    """
    """

    def __init__(self, model_name: str, host: str, port: int, verbose: bool = True):
        """
        Initializes the Generator, connects to Ollama service.
        Args:
            model_name (str): The name of the LLM model to use.
            host (str): The host address of the Ollama server.
            port (int): The port number of the Ollama server.
            verbose (bool): Whether to print verbose output.
        """
        self._model_name = model_name
        self._host = host
        self._port = port
        self._verbose = verbose

        url = host if host and host.startswith(('http', 'https')) else f'http://{host}:{port}'
        self._llm_client = ollama.Client(host=url)

        vprint(self._verbose, f"--> [Generator] RAG System connected to Ollama: {self._llm_client._client.base_url}")
        print(f"--> [Generator] RAG System connected to Ollama: {self._llm_client._client.base_url}")

    def query(self, prompt: str) -> str:
        """
        Processes a user query by generating a response using the LLM.
        Args:
            prompt (str): The prompt to send to the LLM.
        Returns:
            str: The generated response from the LLM.
        """
        try:
            
            vprint(self._verbose, f"--> [Generator] RAG System connected to Ollama: {self._llm_client._client.base_url}")
            response = self._llm_client.chat(
                model=self._model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            answer = response['message']['content']
        except Exception as e:
            vprint(self._verbose, f"--> [Generator] Error connecting to Ollama: {e}")
            return '{"error": "Could not generate response."}'

        vprint(self._verbose, "--> [Generator] Response generated successfully.")
        return answer
        