import logging
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
# from IPython.display import Markdown, display
from llama_index.node_parser import SentenceSplitter
from embedding_manager import Embeddings
import chromadb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def read_data(self):
        logger.info("Reading data from files: %s", self.file_paths)
        # automatically selects the best file reader based on the file extensions
        data_loader = SimpleDirectoryReader(input_files=self.file_paths)
        return data_loader.load_data()

    def chunk_data(self, data, chunk_size=500, chunk_overlap=50):
        logger.info("Parsing data")
        node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" ",
            paragraph_separator="\n\n\n",
            secondary_chunking_regex="[^,.;。]+[,.;。]?"
        )
        return node_parser.get_nodes_from_documents(data)


class DatabaseManager:
    def __init__(self, db_path, collection_name):
        self.db_path = db_path
        self.collection_name = collection_name

    def get_db(self):
        logger.info("Initializing the database at path: %s", self.db_path)
        db = chromadb.PersistentClient(path=self.db_path)
        collection = db.get_or_create_collection(self.collection_name)
        return collection


# class VectorIndexer:
#     def __init__(self, nodes, vector_store, embedding_model, llm_model, indexid, index_path):
#         self.nodes = nodes
#         self.vector_store = vector_store
#         self.embedding_model = embedding_model
#         self.llm_model = llm_model
#         self.indexid = indexid
#         self.index_path = index_path
#         # self.service_context

#     def get_index(self):
#         try:
#             logger.info(f"Load {self.indexid} from local path {self.index_path}")
#             storage_context = StorageContext.from_defaults(vector_store=self.vector_store,
#                                                           persist_dir=self.index_path)
#             index = load_index_from_storage(storage_context=storage_context, index_id=indexid)
            
#         except Exception as e:
#             logger.info("Creating the vector index")
#             storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
#             service_context = ServiceContext.from_defaults(embed_model=self.embedding_model, llm=self.llm_model)
#             index =  VectorStoreIndex(
#                 self.nodes, storage_context=storage_context, service_context=service_context
#             )
#             index.set_index_id(self.indexid)
#             index.storage_context.persist(persist_dir=self.index_path)
#         return index



