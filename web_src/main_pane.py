import streamlit as st
import numpy as np


#---------------------------------------------------------

#>> Llama-Index
from llama_index.evaluation import DatasetGenerator, RelevancyEvaluator
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    LLMPredictor,
    Response,
    PromptHelper,
    download_loader
)

from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.node_parser import SentenceSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.prompts.base import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.llms import ChatMessage, MessageRole


from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer
)

from web_reader import DirectoryWebPageReader
from inference_engine_handler import InferenceEngine_Handler
from retrieval_handler import Retrieval_Handler
import torch

device = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"


chat_role2tag = {
    'system' : '### System:\n',
    'user' : '### User:\n',
    'assistant' : '### Assistant:\n'
}



@st.cache_resource(show_spinner=False)
def setup_chat_environment():
    with st.spinner(text="Loading and indexing the data - hang tight! This should take 1-2 minutes."):
        

        print("Setting up the environment ...")
        # ---------------------------------------------------------------------------
        ## >> ENCODER MODEL
        print("Loading the models ...")

        encoder_model_name = "nickprock/sentence-bert-base-italian-uncased"
        model_name = 'upstage/SOLAR-10.7B-Instruct-v1.0'
        
        ## >> INFERENCE MODEL
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda:0', torch_dtype=torch.float16, cache_dir='cache/')
        tokenizer = AutoTokenizer.from_pretrained(model_name)


        # Load vector database
        print("Loading data ...")

        inf_engine_config = {
            'model' : model,
            'tokenizer' : tokenizer,
            'embedder_model_name' : encoder_model_name,
            'use_cpp' : False
        }
        engine_handler = InferenceEngine_Handler(**inf_engine_config)

        engine = engine_handler
        service_context = engine_handler.service_context

    

        # ---------------------------------------------------------------------------
        # Load documents
        vector_db_path = 'vector_database/'

        retrieval_handler = Retrieval_Handler()
        db_config = {
            'load_from_disk' : True,
            'service_context' : service_context,
            'chromadb_path' : vector_db_path,
            'chunk_size' : 256,
            'documents' : None
        }

        vector_index = retrieval_handler.setup_repository_engine(**db_config)

        
        return vector_index



# ----------------------------------------------------------

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")


context_prompt = """
Le informazioni di contesto sono riportate a seguire.
---------------------
{context_str}
---------------------
"""

system_prompt = f"""
{chat_role2tag['system']}
Sei un chatbot per il supporto alla consultazione dei regolamenti
dell'ateneo di Bologna. Quando formuli una risposta, segui attentamente
queste regole:
- rispondi alle domande poste da studenti e docenti in maniera corretta attenendoti alle informazioni di contesto
fornite;
- non utilizzare la tua conoscenza pregressa su domande tecniche e specifiche sulle regole del dipartimento; 
- non iniziare la tua rispota con "Come Chatbot, .." oppure "Sono un assistente AI .." o cose del genere. Rispondi in maniera diretta;
- sei libero di rispondere autonomamente se non conosci la risposta a domande generiche come ad esempio 'Quando è nata l'Università di Bologna?';
- se non conosci la risposta, rispondi che non sei in grado di fornire informazioni adeguate;
- rispondi sempre in italiano.
"""
system_prompt_msg = ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)

index = setup_chat_environment()

NUM_RESOURCES_IN_CONTEXT = 4
chat_engine = index.as_chat_engine(chat_mode="context", 
                                   verbose=False, 
                                   similarity_top_k=NUM_RESOURCES_IN_CONTEXT,
                                   context_template=context_prompt,
                                   prefix_messages=[system_prompt_msg])



# Initialize Chat message history
st.header("UNIBO Chat - beta")


if "messages" not in st.session_state.keys(): 
    st.session_state.messages = [
        {"role": "assistant", "content": "Ciao! Sono qui se hai qualche domanda ..."}
    ]

# Prompt for user input and save to chat history
if prompt := st.chat_input("La tua domanda ..."): 
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display the prior chat messages
for message in st.session_state.messages: 
    _avatar="/CA_Synthetic/web_app/images/logo-unibo.png" if message["role"] == 'assistant' else None
    _name = "UNIBO AI assistant" if message["role"] == 'assistant' else "user"
    with st.chat_message(_name, avatar=_avatar):
        st.write(message["content"])


# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar="/CA_Synthetic/web_app/images/logo-unibo.png"):
        with st.spinner("Ci sto pensando su..."):

            response = chat_engine.chat(chat_role2tag['user'] + prompt)       
            out_response = response.response

            # Remove spurious lines
            reps = ['Reply:', "La risposta che ho ricevuto dall'AI è:", "Response", 'Risposta:', ':\n', ':', 'Ho iniziato la mia risposta autonomamente)\n'] 
            for rep in reps:
                if rep in out_response:
                    out_response = out_response.split(rep)[1].strip()
            
            out_response.replace('"','')

            # Save and write out the answer
            st.write(out_response)

            message = {"role": "assistant", "content": out_response}

            st.session_state.messages.append(message) # Add response to message history
