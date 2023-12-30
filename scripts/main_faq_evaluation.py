import logging
import os
import sys
import numpy as np
import json
import math
import wandb

import datasets
from datasets import (
    load_dataset, 
    load_from_disk
)

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EvalPrediction,
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed, 
    get_scheduler,
)

from transformers.trainer_utils import get_last_checkpoint

import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader


from web_src.data_classes import (
    DataTrainingArguments,
    ModelArguments
)

from src.data_retriever import (
    RetrieverEngine
)

from web_src.zero_shot_model_evalautor import EvaluationPipeline_Handler

def main():

    parser = HfArgumentParser((
        ModelArguments, 
        DataTrainingArguments, 
        TrainingArguments           # predefined in transformers
    ))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    # ----------------------------------------------------
    # >> LOGGER
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    #logging.getLogger("codecarbon").setLevel(logging.ERROR)

    if training_args.should_log:
        # The default of training_args.log_level is passive, 
        # so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()

    logger = logging.getLogger(__name__)

    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


    if training_args.output_dir is None:
        training_args.output_dir = data_args.task_name + "_" + model_args.model_name_or_path

    # Set checpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed 
    set_seed(training_args.seed)

    print("\n"*5)
    print("-"*100)

    # INITIALIZE GENERATIVE MODEL
    if model_args.use_retrieval:
        print("Loading Generator model ...")
        generation_config = {
            'training_args' : training_args,
            'model_args' : model_args,
            'data_args' : data_args
        }
        data_generator = SyntheticGenerator(**generation_config)
        service_context = data_generator.generator_handler

        # INITIATE RETRIEVAL DIRECTORY
        retriever_config = {
            "data_path" : data_args.data_repository_path,
            "is_directory" : data_args.retrieval_repo_is_directory
        }
        retriever = RetrieverEngine(**retriever_config)

        print("Load and process retrieval data ...")
        documents = retriever.get_documents()

        retr_engine_config = {
            "documents" : documents,
            "service_context" : service_context,
            "chunk_size" : model_args.chunk_size
        }
        
        retriever.setup_repository_engine(**retr_engine_config)
        query_engine = retriever.get_query_engine()
    else:
        # Zero-shot inference
        evalutor_config = {
            'model_name' : model_args.model_name_or_path,
            'tokenizer_name' : model_args.tokenizer_name,
            'language' : data_args.lang
        }
        evaluator = EvaluationPipeline_Handler(**evalutor_config)

        # load model
        print("Loading pretrained model for generation ...")
        model_loading_config = {
            'model_args' : model_args,
            'training_args' : training_args,
            'data_args' : data_args,
            'use_vllm' : model_args.use_vllm_engine
        }
        evaluator.load_mode(**model_loading_config)

        # load data
        print("Loading FAQ dataset from path ...")
        with open(data_args.data_path,'r') as f_in:
            faq_data = json.load(f_in)

        # Start evalutation
        print("Starting generation and evaluation ...")
        statistiscs = evaluator.run_evaluation(faq_data)

        print('\n'*2)
        print('-'*50)
        print(statistiscs)
        print('\n'*2)
        
        with open(training_args.output_dir, 'w') as f_out:
            json.dump(statistiscs, f_out)
    

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()