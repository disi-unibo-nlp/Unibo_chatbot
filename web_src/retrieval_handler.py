from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding


from tqdm import tqdm

# Chroma DB
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb


class Retrieval_Handler():

    def __init__(self):
        super(Retrieval_Handler, self).__init__()

    
    def get_retriever_engine(self,
                             top_k_similarity : int = 3):

        return self.vector_index.as_retriever(similarity_top_k=top_k_similarity)
    
    def get_query_engine(self,
                        num_resources : int = 5):
        return self.vector_index.as_query_engine(similarity_top_k=num_resources)

    def __setup_chromaDb_repo(self,
                              service_context,
                              documents = None,
                              chunk_size : int = 512,
                              chunk_overlap : int = 0,
                              load_from_disk : bool = False,
                              chromadb_path : str = 'chromadb'):
        """
        Initialize a ChromaDB database.
        """

        db = chromadb.PersistentClient(path=chromadb_path)
        
        # Create persistent (on disk) database - or load it if it exists already
        chroma_collection = db.get_or_create_collection("quickstart")
        
        if not load_from_disk:
            print("[LOG] Creating ChromaDB and storing to disk")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            index = VectorStoreIndex.from_documents(
                documents, 
                storage_context=storage_context, 
                service_context=service_context,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            print("[LOG] Loading ChromaDB from disk")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

            index = VectorStoreIndex.from_vector_store(
                vector_store,
                service_context=service_context,
            )

        self.vector_index = index

        return index

    def setup_repository_engine(self,
                                documents,
                                service_context : ServiceContext,
                                add_new_documents_to_existing_db : bool = False,
                                load_from_disk : bool = False,
                                chromadb_path : str = 'data/chromadb',
                                chunk_overlap : int = 0,
                                chunk_size : int = 512):

        service_context.transformations[0].include_metadata = False # DO NOT include resources' metadata in the context

        # set up ChromaVectorStore and load in data
        chromadb_config = {
            'service_context' : service_context,
            'documents' : documents,
            'load_from_disk' : load_from_disk,
            'chromadb_path' : chromadb_path,
            'chunk_size' : chunk_size,
            'chunk_overlap' : chunk_overlap
        }
        index = self.__setup_chromaDb_repo(**chromadb_config)

        if add_new_documents_to_existing_db:
            for doc in tqdm(documents, desc="[LOG] Adding documents to existing Chroma database ..."):
                index_config = {
                    'document' : doc,
                    'service_context' : service_context
                }
                index.insert(**index_config)


        return index