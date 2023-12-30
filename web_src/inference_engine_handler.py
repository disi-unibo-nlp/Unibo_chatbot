import logging
import sys
import pandas as pd
import json
import torch
from tqdm import tqdm

import random

# >>> Llama-Index

from llama_index.evaluation import DatasetGenerator, RelevancyEvaluator
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    LLMPredictor,
    Response,
    PromptHelper
)


from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.node_parser import SentenceSplitter

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.prompts.base import PromptTemplate


from llama_index.llms import HuggingFaceLLM
#from src.quantization_config import QuantizationConfig
#from src.generation_utils import vLLM_Wrapper
#from vllm import LLM, SamplingParams

class ConfigurationDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class InferenceEngine_Handler():

    def __init__(self,
                model,
                tokenizer,
                embedder_model_name : str,
                use_cpp : bool = False,
                max_seq_length : int = 512, 
                context_window : int = 4096,
                max_new_tokens : int = 256,
                chunk_size : int = 512,
                chunk_overlap : int = 0, 
                model_name_or_path : str = None,
                tokenizer_name : str = None,
                quantize_4_bits : bool = False,
                chunk_overlap_ratio : bool = 0.2,
                temperature : float = 0.8,
                top_k : float = 10,
                top_p : int = 0.8,
                enable_sampling : bool = True,
                seed : int = 42):

        super(InferenceEngine_Handler, self).__init__()

        self.model = None
        self.encoder = None
        self.prompt_helper = None
        self.service_context = None

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Generation config
        gen_config = {
            'temperature' : temperature,
            'top_k' : top_k,
            'top_p' : top_p,
            'enable_sampling' : enable_sampling
        }
        gen_config = ConfigurationDict(gen_config)


        # Core model config
        model_config = {
            'model_name_or_path' : model_name_or_path,
            'embedder_model_name' : embedder_model_name,
            'model' : model,
            'tokenizer' : tokenizer,
            'tokenizer_name' : tokenizer_name,
            'max_seq_length' : max_seq_length, 
            'context_window' : context_window,
            'max_new_tokens' : max_new_tokens,
            'quantize_4_bits' : quantize_4_bits,
            'chunk_overlap_ratio' : chunk_overlap_ratio
        }
        model_config = ConfigurationDict(model_config)


        print("Initializing the inference engine ...")
        self.__init_models(engine_config=model_config,
                           genereation_config=gen_config,
                           use_cpp=use_cpp,
                           seed=seed)


        print("Initializing the Service Context ...")
        self.__initialize_service_context(engine_config=model_config)

        print(f"USE CPP --> {use_cpp}")



    @property  
    def generator_handler(self): 
        return self.service_context


    def __init_models(self,
                      engine_config : ConfigurationDict, 
                      genereation_config : ConfigurationDict,
                      use_cpp : bool = False,
                      seed : int = 42):
        """
        Initialize the inference engine and the encoding modules.
        """
        
        model_name = engine_config.model_name_or_path 
        device = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"

        # Quantization
        self.quantization_config = None
        if engine_config.quantize_4_bits:
            quant_config = QuantizationConfig()
            self.quantization_config = quant_config.quantizatiom_configuration


        #if not model_args.use_vllm_engine:
        if not use_cpp:
            self.model = HuggingFaceLLM(
                context_window=engine_config.context_window,
                max_new_tokens=engine_config.max_new_tokens,
                model=engine_config.model,
                tokenizer=engine_config.tokenizer,
                #tokenizer_name=engine_config.tokenizer_name,
                #model_name=model_name,
                device_map=device,
                tokenizer_kwargs={
                    "max_length" : engine_config.max_seq_length,
                    "cache_dir" : "cache/"
                },
                model_kwargs={
                    "torch_dtype": torch.float16,
                    "cache_dir" : "cache/",
                    "use_flash_attention_2" : True,
                    #"load_in_4bit" : True,
                    "attn_implementation" : "flash_attention_2"
                    #"quantization_config" : self.quantization_config
                },
                generate_kwargs={
                    "temperature": genereation_config.temperature, 
                    "top_k" : genereation_config.top_k,
                    "top_p" : genereation_config.top_p,
                    "do_sample": genereation_config.enable_sampling
                }
            )
        else:
            cpp_model = LlamaCPP(
                model_path = model_name,
                context_window=engine_config.context_window,
                max_new_tokens=engine_config.max_new_tokens,
                model_kwargs={
                    "torch_dtype": torch.float16,
                    "cache_dir" : "cache/",
                    "use_flash_attention_2" : True,
                    "device" : device,
                    "n_gpu_layers" : 40,
                    "attn_implementation" : "flash_attention_2"
                },
                generate_kwargs={
                    "temperature": genereation_config.temperature, 
                    "top_k" : genereation_config.top_k,
                    "top_p" : genereation_config.top_p
                }
            )

            #self.model = LLMPredictor(llm=cpp_model)
            self.model = cpp_model


        # else:
        #     # Load the LLM
        #     generate_kwargs = {
        #         "temperature": genereation_config.temperature, 
        #         "top_k" : genereation_config.top_k,
        #         "top_p" : genereation_config.top_p,
        #         "max_tokens" : engine_config.max_new_tokens
        #     }
        #     sampling_params = SamplingParams(**generate_kwargs)

        #     vllm_model = LLM(
        #         model=model_name,
        #         tokenizer=engine_config.tokenizer_name,
        #         quantization='awq',
        #         dtype='half',
        #         gpu_memory_utilization=.95,
        #         max_model_len=4096,
        #         seed=seed
        #     )

        #     self.model = vLLM_Wrapper(
        #         context_window=engine_config.context_window,
        #         max_new_tokens=engine_config.max_new_tokens,
        #         tokenizer_name=engine_config.tokenizer_name,
        #         model=vllm_model,
        #         device_map=device,
        #         tokenizer_kwargs={
        #             "max_length" : engine_config.max_seq_length,
        #             "cache_dir" : "cache/"
        #         },
        #         generate_kwargs=sampling_params
        #     )


        encoder_model_name = engine_config.embedder_model_name
        if encoder_model_name != 'local':
            self.encoder = LangchainEmbedding(
                HuggingFaceEmbeddings(
                    model_name=encoder_model_name,
                    model_kwargs = {'device': device}
                ),
                embed_batch_size=2
            )
        else:
            self.encoder = "local" # use model's encoder to embedd resources


    def __initialize_service_context(self,
                                     engine_config : ConfigurationDict):
        """
        Initialize the Service Context.
        """

        prompt_config = {
            "context_window" : engine_config.context_window,
            "num_output" : engine_config.max_new_tokens,
            "chunk_overlap_ratio" : engine_config.chunk_overlap_ratio
        }
        self.prompt_helper = PromptHelper(**prompt_config)


        system_prompt = """
        Sei un esperto dei regolamenti dell'Università di Bologna. Il tuo lavoro è quello di rispondere
        a domande tecniche.
        Segui le seguenti regole:
        - rispondi in modo corretto;
        - non aggiungere informazioni non inerenti con la richiesta;
        - rispondi sempre in italiano.
        """

        service_context_config = {
            "llm" : self.model,
            "embed_model" : self.encoder,
            "prompt_helper" : self.prompt_helper,
            "chunk_size" : self.chunk_size,
            "chunk_overlap" : self.chunk_overlap
            #"system_prompt" : system_prompt
        }

        self.service_context = ServiceContext.from_defaults(**service_context_config)
