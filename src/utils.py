import os
import yaml
import glob
from project_dirs import PROJECT_DIR, DATA_DIR
from data_loader import DatabaseManager, DataLoader, VectorIndexer
from embedding_manager import Embeddings
from llama_index.vector_stores import ChromaVectorStore


def load_config(cnf_dir=PROJECT_DIR, cnf_name="config.yml"):
    """
    load the yaml file
    """
    config_file = open(os.path.join(cnf_dir, cnf_name))
    return yaml.load(config_file, yaml.FullLoader)


def list_all_filepaths(
    common_dir: str,
    folder: str,
    extension: str,
):
    path = os.path.join(common_dir, folder, f"*{extension}")
    filenames = glob.glob(path)

    if not filenames:
        path = os.path.join(
            common_dir, folder, "**\\", f"*{extension}"
        )  # search into subdirectories
        filenames = glob.glob(path)

    return filenames


def list_all_filepaths_for_list_of_extentions(
    extentions = None, common_dir: str = DATA_DIR, folder: str = ''
) -> list:
    """list all filepaths for a list of extentions.

    Args:
        extentions (iter of str): list of extentions
        common_dir (str): common directory
        folder (str): folder in the common directory

    Returns:
        list: list of filepaths
    """
    if not extentions:
        extentions = ['pdf', 'txt', 'xlsx']
    filepaths = []
    for ext in extentions:
        filepaths.append(
            list_all_filepaths(
                common_dir=common_dir,
                folder=folder,
                extension=ext,
            )
        )
    flatten_filepaths_list = [path for list in filepaths for path in list]
    return flatten_filepaths_list


# def load_data_to_vector_db(
#     data_path, 
#     db_path, 
#     db_collection_name, 
#     embedding_mode, 
#     local_model_name,
#     chunk_size=500, 
#     chunk_overlap=50,
#     ):

#     # Load data files
#     file_paths = list_all_filepaths_for_list_of_extentions(common_dir = data_path)
#     data_loader = DataLoader(file_paths=file_paths)
#     data = data_loader.read_data()
#     chunks = data_loader.chunk_data(data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

#     # Database 
#     db_manager = DatabaseManager(db_path=db_path, collection_name=db_collection_name)
#     db_collection = db_manager.initialize_db()

#     # Embedding Model 
#     embedding_model = Embeddings(embedding_mode, local_model_name)

#     # Vector Store and Index 
#     vector_store = ChromaVectorStore(chroma_collection=db_collection)
#     indexer = VectorIndexer(
#         nodes=chunks, 
#         vector_store=vector_store, 
#         embedding_model=embedding_model.embedding_model,
#         llm_model = 
#     )
#     index = indexer.create_index()


