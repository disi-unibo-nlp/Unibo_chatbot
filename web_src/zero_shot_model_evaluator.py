import torch
import numpy as np

from vllm import LLM, SamplingParams
from transformers import (
    Trainer,
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoConfig
)


from web_src.prompts import ZERO_SHOT_PROMPTS_BY_LANG



class EvaluationPipeline_Handler():

    def __init__(self,
                 model_name,
                 tokenizer_name,
                 use_vllm : bool = False,
                 language : str = 'it'):
        super(EvaluationPipeline_Handler, self).__init__()

        self._lang = language
        self._query_prompt = ZERO_SHOT_PROMPTS_BY_LANG[language]

        self._model_name = model_name
        self._tokenizer_name = tokenizer_name
        self.model = None
        self.tokenizer = tokenizer
        
        self.use_vllm = use_vllm

        self.quantization_config = None


    def load_model(self,
                   model_args,
                   training_args,
                   data_args):
        
        device = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"
        model_name = self._model_name

        if not self.use_vllm:
            self.model = None


            self._tokenizer_config = {
                'return_tensors' : 'pt',
                'max_length' : data_args.max_seq_length,
                'padding' : 'max_length',
                'truncation' : True
            }         
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            model_config = AutoConfig(model_name)
            model_config = {
                'pretrained_model_name_or_path' : model_namem,
                'config' : model_config,
                'cache_dir' : 'cache/',
                'torch_dtype' : torch.float16 if training_args.fp16 else torch.float32
            }
            self.model = AutoModelForCausalLM.from_pretrained(**model_config).to(device)

            self.sampling_params={
                "temperature": model_args.temperature, 
                "top_k" : model_args.top_k,
                "top_p" : model_args.top_p,
                "do_sample": model_args.enable_sampling,
                "max_length" : model_args.max_new_tokens
            }
            
        else:
            # Load the LLM
            generate_kwargs = {
                "temperature": model_args.temperature, 
                "top_k" : model_args.top_k,
                "top_p" : model_args.top_p,
                "max_tokens" : model_args.max_new_tokens
            }
            sampling_params_vllm = SamplingParams(**generate_kwargs) if model_args.enable_sampling else None
            self.sampling_params = {
                'prompts' : None,
                'sampling_params' : sampling_params_vllm,
                'use_tqdm' : True
            }

            self.model = LLM(
                model=model_name,
                tokenizer=self._tokenizer_name,
                quantization='awq',
                dtype='half',
                gpu_memory_utilization=.95,
                max_model_len=4096,
                seed=training_args.seed
            )

    def __prompt_query(self,
                       question : str,
                       options : str):
        
        result = self._query_prompt.format(question, options)

        return result

    def __generate(self, questions):

        if self.use_vllm:
            generation_config = self.generation_config
            generation_config['prompts'] = questions
        else:
            # Explicitely tokenize data before generation
            input_ids = self.tokenizer(questions, **self._tokenizer_config).input_ids
            generation_config = self.generation_config
            generation_config['input_ids'] = input_ids

        # Generate results
        result = self.llm.generate(**generation_config)

        # Parse answers
        answers = []
        for output in results:
            if self.use_vllm:
                answer = output.outputs[0].text
            else:
                answer = ...

                
            try:
                answer_letter = answer.split(')')[0].split('(')[1].strip()
            except:
                # No desired format
                answer_letter = answer.strip()

            answers.append(answer_letter)

        return answers
        
    def __evaluate(self,
                  predictions : list,
                  labels : list):
        
        # Accuaracy
        precision = [pred == label for pred, label in zip(predictions, labels)]
        acc = np.count_nonzero(precision)

        return acc



    def run_evaluation(self,
                       faq_data : dict):

        lang_faqs = faq_data[self._lang]

        # Prompt input queries
        answers = lang_faqs['Answer']
        prompted_query = list(map(lambda x : self.__prompt_query(x['Question'], x['Options']), lang_faqs))

        # Generate answers
        predictions = self.__generate(prompted_query)

        # Evaluate output
        accuracy = self.__evaluate(predictions, answers)


        return {'accuracy' : accuracy}


        
