from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding

from web_reader import DirectoryWebPageReader
from inference_engine_handler import InferenceEngine_Handler
from retrieval_handler import Retrieval_Handler

from typing import List
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer
)
from peft import PeftModel


class VectorDB_Generator():

    def __init__(self,
                 model : str,
                 tokenizer : str,
                 documents_encoder_model_name : str,
                 vector_db_out_path : str,
                 chunk_size : int = 512,
                 use_cpp : bool = False):

        super(VectorDB_Generator, self).__init__()

        self.output_path = vector_db_out_path
        self.use_cpp = use_cpp

        self.engine = None
        self.service_context = None
        self.vector_index = None
        self.retrieval_handler = None

        print("Setting up the inference environment ...")
        setup_config = {
            'model' : model,
            'embed_model_name' : documents_encoder_model_name,
            'tokenizer' : tokenizer,
            'chunk_size' : chunk_size
        }
        self.__setup_environment(**setup_config)


    def __setup_environment(self,
                            model,
                            tokenizer,
                            embed_model_name : str,
                            chunk_size : int = 512,
                            inference_model_name : str = None,
                            tokenizer_name : str = None):
        """
        Create the inference environment by defining the Service Context.
        """

        # Create Service Context
        inf_engine_config = {
            'model' : model,
            'tokenizer' : tokenizer,
            'model_name_or_path' : inference_model_name,
            'embedder_model_name' : embed_model_name,
            'tokenizer_name' : tokenizer_name,
            'chunk_size' : chunk_size,
            'use_cpp' : self.use_cpp
        }
        engine_handler = InferenceEngine_Handler(**inf_engine_config)

        self.engine = engine_handler
        self.service_context = engine_handler.service_context


    def __load_documents(self,
                         documents_path : str,
                         file_names_to_exclude : List[str] = None):
        """
        Load document nodes from the given path.
        """
        
        reader_config = {
            'input_dir' : documents_path,
            'html_to_text' : True,
            'recursive' : True,
            'exclude' : file_names_to_exclude
        }
        dir_reader = DirectoryWebPageReader(**reader_config)

        documents = dir_reader.load_data(show_progress=True, ignore_links=True)

        return documents

    
    def create_vector_db(self,
                         data_path : str,
                         load_from_disk : bool = False,
                         add_new_documents_to_existing_db : bool = False,
                         file_names_to_exclude : List[str] = None,
                         chunk_size : int = 256,
                         chunk_overlap : int = 0):

        documents = None
        if not load_from_disk or add_new_documents_to_existing_db:
            print("Dividing data into chunks ...")
            documents = self.__load_documents(documents_path = data_path, file_names_to_exclude=file_names_to_exclude)

        print("Indexing the dataset...")
        self.retrieval_handler = Retrieval_Handler()

        db_config = {
            'documents' : documents,
            'load_from_disk' : load_from_disk,
            'add_new_documents_to_existing_db' : add_new_documents_to_existing_db,
            'service_context' : self.service_context,
            'chromadb_path' : self.output_path,
            'chunk_size' : chunk_size,
            'chunk_overlap' : chunk_overlap
        }

        self.vector_index = self.retrieval_handler.setup_repository_engine(**db_config)

        return self.vector_index

    def get_query_engine(self,num_resources : int = 5):
        return self.retrieval_handler.get_query_engine(num_resources)



# ------------------------------------------
# Model configuration

encoder_model_name = "nickprock/sentence-bert-base-italian-uncased"

model_name = "upstage/SOLAR-10.7B-Instruct-v1.0"
tokenizer_name = "upstage/SOLAR-10.7B-Instruct-v1.0"
DB_OUT_PATH = 'vector_database/'


# --------------------------------------

USE_CPP = False
if USE_CPP:
    model_file = 'models/solar_gguf/solar-10.7b-instruct-v1.0.Q4_K_M.gguf'
    model = model_file
    tokenizer = None
else:
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda:0', torch_dtype=torch.float16, cache_dir='cache/')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


CHUNK_SIZE = 256

generator_config  = {
    'model' : model,
    'tokenizer' : tokenizer,
    'documents_encoder_model_name' : encoder_model_name,
    'vector_db_out_path' : DB_OUT_PATH,
    'chunk_size' : CHUNK_SIZE,
    'use_cpp' : USE_CPP
}
db_generator = VectorDB_Generator(**generator_config)

DATA_PATHs = [
    'data/unibo_web/unibo.sitoweb.it',
    'data/unibo_web/disi.unibo.it',
    'data/unibo_web/corsi.unibo.it']

db_config = {
    'data_path' : DATA_PATHs[2],
    'chunk_size' : CHUNK_SIZE,
    'chunk_overlap' : 0,
    'load_from_disk' : True,
    'add_new_documents_to_existing_db' : False
}
vector_index = db_generator.create_vector_db(**db_config)



# ---------------
from llama_index.prompts import PromptTemplate


NUM_RESOURCES = 6
query_engine = db_generator.get_query_engine(num_resources=NUM_RESOURCES)


qa_prompt = """
Sei un assistente che risponde alle domande poste sui regolamenti dell'università di Bologna.
Usa le informazioni riportate nei paragrafi estratti per rispondere alla domanda.
Se non conosci la risposta, rispondi semplicemente che non sai rispondere.

Segui le seguenti regole:
- rispondi in modo corretto;
- non aggiungere informazioni non inerenti con la richiesta;
- rispondi sempre in italiano.

---------------------
Di seguito si riportano le informazioni di contesto:
### Contesto:
{context_str}

---------------------
### Domanda:
{query_str}

---------------------
### Risposta:
"""

qa_prompt = PromptTemplate(qa_prompt)

query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": qa_prompt}
)

TEST_QUESTION = "A proposito dello svolgimento dell'attivià di tirocinio per il mio piano di studi, ho trovato un'azienda che mi interessa ma non è convenzionata con l'Ateneo di Bologna. Posso comunque svolgere il tirocinio presso di loro o è necessario avviare una pratica specifica?"
# TRUE_ANSWER = "Si ma solo se prima l’Ente si convenziona con l’Ateneo collegandosi a tirocini.unibo.it e seguendo la procedura guidata fino alla fine."

gen_out = query_engine.query(TEST_QUESTION)

print("\n"*3)
print(gen_out.response.split('Note')[0].strip())


resources = '--------\n\n'.join([n.text for n in gen_out.source_nodes])
with open('results/res_parsed.txt','w') as f_out: f_out.write(resources)

