import logging
import os
from llama_index.llms import LlamaCPP, OpenAI, AzureOpenAI
from llama_index.llms.llama_utils import completion_to_prompt,messages_to_prompt

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY', '')
AZURE_ENDPOINT = os.environ.get('AZURE_ENDPOINT', '')
AZURE_API_VERSION = os.environ.get('AZURE_API_VERSION', '')
AZURE_DEPLOYMENT_NAME = os.environ.get('AZURE_DEPLOYMENT_NAME', '')
AZURE_LLM_MODEL = os.environ.get('AZURE_LLM_MODEL', '')

logger = logging.getLogger(__name__)

class LLMMain:
    def __init__(self, llm_mode, llm_model_path=None) -> None:
        self.llm_mode = llm_mode
        self.llm_model_path = llm_model_path
        self.llm = None  
        logger.info("Initializing the LLM in mode=%s", self.llm_mode)
        self._get_llm()

    def _get_llm(self):
        if self.llm_mode == "local":
            self.llm = LlamaCPP(
                model_url=None,
                model_path=self.llm_model_path,
                temperature=0.0,
                max_new_tokens=256,
                context_window=3900,
                generate_kwargs={},
                messages_to_prompt=messages_to_prompt,
                completion_to_prompt=completion_to_prompt
            )
        elif self.llm_mode == "openai":
            self.llm = OpenAI(temperature=0.0, api_key=OPENAI_API_KEY)
            
        elif self.llm_mode == "azureopenai":
            self.llm = AzureOpenAI(
                model=AZURE_LLM_MODEL,
                deployment_name=AZURE_DEPLOYMENT_NAME,
                api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_ENDPOINT,
                api_version=AZURE_API_VERSION,
            )
            
        logger.info("Got LLM model %s", self.llm)
