import logging
import os
from llama_index.embeddings import HuggingFaceEmbedding, AzureOpenAIEmbedding
from llama_index import OpenAIEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

import logging
import os
from llama_index.embeddings import HuggingFaceEmbedding, AzureOpenAIEmbedding
from llama_index import OpenAIEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Embeddings:
    def __init__(self, embedding_mode="openai", local_model_name=None):
        self.embedding_mode = embedding_mode
        self.local_model_name = local_model_name

        logger.info("Getting the %s embedding model %s", embedding_mode, local_model_name)
        if embedding_mode == "local":
            self.embedding_model = self._get_local_embedding()

        elif embedding_mode == "openai":
            self.embedding_model = self._get_openai_embedding()
            
        elif embedding_mode == "azureopenai":
            self.embedding_model = self._get_azureopenai_embedding()
            
        logger.info("Created embedding model %s", self.embedding_model)

    def _get_local_embedding(self):
        return HuggingFaceEmbedding(model_name=self.local_model_name)

    def _get_openai_embedding(self):
        OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
        return OpenAIEmbedding(api_key=OPENAI_API_KEY)
    
    def _get_azureopenai_embedding(self):
        AZURE_EMBEDDING_MODEL = os.environ.get('AZURE_EMBEDDING_MODEL', '')
        AZURE_EMBEDDING_DEPLOYMENT_NAME = os.environ.get('AZURE_EMBEDDING_DEPLOYMENT_NAME', '')
        AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY', '')
        AZURE_ENDPOINT = os.environ.get('AZURE_ENDPOINT', '')
        AZURE_API_VERSION = os.environ.get('AZURE_API_VERSION', '')
        
        return AzureOpenAIEmbedding(
            model=AZURE_EMBEDDING_MODEL,
            deployment_name=AZURE_EMBEDDING_DEPLOYMENT_NAME,
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
        )
        
AZURE_EMBEDDING_MODEL = os.environ.get('AZURE_EMBEDDING_MODEL', '')
AZURE_EMBEDDING_DEPLOYMENT_NAME = os.environ.get('AZURE_EMBEDDING_DEPLOYMENT_NAME', '')
AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY', '')
AZURE_ENDPOINT = os.environ.get('AZURE_ENDPOINT', '')
AZURE_API_VERSION = os.environ.get('AZURE_API_VERSION', '')

class Embeddings:
    def __init__(self, embedding_mode="openai", local_model_name=None):
        self.embedding_mode = embedding_mode
        self.local_model_name = local_model_name

        logger.info("Getting the %s embedding model %s", embedding_mode, local_model_name)
        if embedding_mode == "local":
            self.embedding_model = self._get_local_embedding()

        elif embedding_mode == "openai":
            self.embedding_model = self._get_openai_embedding()
            
        elif embedding_mode == "azureopenai":
            self.embedding_model = self._get_azureopenai_embedding()
            
        logger.info("Created embedding model %s", self.embedding_model)

    def _get_local_embedding(self):
        return HuggingFaceEmbedding(model_name=self.local_model_name)

    def _get_openai_embedding(self):
        return OpenAIEmbedding(api_key=OPENAI_API_KEY)
    
    def _get_azureopenai_embedding(self):
        return AzureOpenAIEmbedding(
            model=AZURE_EMBEDDING_MODEL,
            deployment_name=AZURE_EMBEDDING_DEPLOYMENT_NAME,
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
        )
        
