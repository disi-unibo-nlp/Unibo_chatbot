import logging
import os
import sys
import numpy as np
import json
import math
import wandb

import nltk
from torch.utils.data import DataLoader
import torch
from torch.nn.functional import softmax
import torch.nn as nn

from datasets import (
    load_dataset, 
    load_from_disk,
    Dataset
)
import evaluate
import transformers
import datasets
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
    BitsAndBytesConfig
)

from trl import (
    DPOTrainer, 
    SFTTrainer,
    DataCollatorForCompletionOnlyLM
)
from trl.trainer import ConstantLengthDataset

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from codecarbon import EmissionsTracker


# Custom libraries

from web_src.data_classes import (
    DataTrainingArguments,
    ModelArguments
)

from src.data_retriever import (
    RetrieverEngine
)

from web_src.prompts import TRAINING_PROMPTS

import nltk
nltk.download('punkt', quiet=True)

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

    # INIT WANDB
    wandb.login(key="26e3177c806f7a09d4b5c02ada52b782adb99009")


    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logging.getLogger("codecarbon").setLevel(logging.ERROR)

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
        training_args.output_dir = 'finetuning' + "_" + model_args.model_name_or_path

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

    def load_data(dict_data_path = None):
        if data_args.dataset_name_local is not None:
            # Loading a local dataset.
            raw_datasets = load_from_disk(data_args.dataset_name_local)
        elif data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                data_args.dataset_name,
                data_args.lang,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        elif data_args.data_from_dict:

            with open(dict_data_path, 'r') as f_in:
                data_dict = json.load(f_in)

            # Create dataset from dict
            raw_datasets = Dataset.from_dict(data_dict[data_args.lang])

            if data_args.split_data:
                raw_datasets = raw_datasets.train_test_split(test_size=data_args.train_test_split_ratio)
            else:
                train_dataset = raw_datasets
                eval_dataset = {}

                raw_datasets = Dataset.from_dict({
                    'train' : train_dataset,
                    'test' : eval_dataset 
                })
        else:
            data_files = {}
            if data_args.train_file is not None:
                data_files["train"] = data_args.train_file
                extension = data_args.train_file.split(".")[-1]
            if data_args.validation_file is not None:
                data_files["validation"] = data_args.validation_file
                extension = data_args.validation_file.split(".")[-1]
            if data_args.test_file is not None:
                data_files["test"] = data_args.test_file
                extension = data_args.test_file.split(".")[-1]
            raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )

        return raw_datasets

    # Load pretrained model and tokenizer

    device_type = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    print(f">>> AVAILABLE CUDA DEVICES: {torch.cuda.device_count()} \nDEVICE: {device}")

    model_config = {
        'pretrained_model_name_or_path' : model_args.model_name_or_path,
        'cache_dir' : 'cache/',
        'torch_dtype' : torch.float16 if training_args.fp16 else torch.float32,
        'device_map' : device
    }
    if model_args.use_peft:
        bnb_4bit_compute_dtype = 'float16' if training_args.fp16 else 'float32'
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

        # bnb_config = BitsAndBytesConfig(
        #     load_in_8bit=True,
        #     bnb_8bit_quant_type="nf8",
        #     bnb_8bit_compute_dtype=compute_dtype,
        # )

        model_config['quantization_config'] = bnb_config


    model = AutoModelForCausalLM.from_pretrained(**model_config)
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
                                              cache_dir='cache/')
    tokenizer.padding_side = 'right'

    print("ALLOCATED MEMORY",round(torch.cuda.memory_allocated(device) / 1000000000, 2), "GB")


    if model_args.use_peft:
        #Define LoRA Config
        lora_config = LoraConfig(
                        r=16,
                        lora_alpha=32,
                        lora_dropout=0.1,
                        bias="none",
                        task_type=TaskType.CAUSAL_LM)
        
        # prepare int-8 model for training
        model = prepare_model_for_int8_training(model, use_gradient_checkpointing=True)

        # add LoRA adaptor
        model = get_peft_model(model, lora_config)


    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    padding = "max_length" if data_args.pad_to_max_length else False

    # Metric
    metric_rouge = evaluate.load("rouge")
    metric_bertscore = evaluate.load("bertscore")

    def compute_metrics(eval_preds):

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # get token ids from logits
        preds = np.argmax(preds, axis=-1)

        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = [pred.strip() for pred in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        decoded_labels = [label.strip() for label in tokenizer.batch_decode(labels, skip_special_tokens=True)]

        # rougeLSum expects newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label)) for label in decoded_labels]

        result = metric_rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 2) for k, v in result.items()}

        result["R"] = round(np.mean([result["rouge1"], result["rouge2"], result["rougeL"]]) / \
            (1 + (np.var([result["rouge1"]/100, result["rouge2"]/100, result["rougeL"]/100]))), 2)
        
        decoded_preds = [pred.replace("\n", " ") for pred in decoded_preds]
        decoded_labels = [label.replace("\n", " ") for label in decoded_labels]

        result_bs = metric_bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang=data_args.lang,
                                             idf=True, rescale_with_baseline=True)# model_type=model_args.model_for_bertscore)
        result["bertscore"] = round(sum(result_bs["f1"]) / len(result_bs["f1"]) * 100, 2)
        result["gen_len"] = np.mean([np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds])

        return result


    # ------------------------------------------------------
    # Training

    # >> Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)
    

    def formatting_prompts_func(examples):
        """
        If SFT training, prompt the input data and store into a unique column 'text'.
        """
        prompt = TRAINING_PROMPTS[data_args.lang]
    
        inputs = []
        for i in range(len(examples['Question'])):
            question = examples['Question'][i]
            answer = examples['Answer'][i]

            prompted_input = prompt.format(question, answer)
            inputs.append(prompted_input)

        return inputs
    

    def start_training(trainer, 
                       checkpoint,
                       train_type : str = 'sft') -> dict:

        output_path = os.path.join(training_args.output_dir, f'_{train_type}') if training_args.output_dir else None

        # Start training with Emission tracking
        train_tracker = EmissionsTracker(measure_power_secs=100000, save_to_file=False)
        train_tracker.start()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        train_emissions = train_tracker.stop()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        print(">>> DONE WITH TRAINING")

        # Store results
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        metrics["train_emissions"] = train_emissions

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


        kwargs = {"finetuned_from": model_args.model_name_or_path}

        if data_args.lang is not None:
            kwargs["language"] = data_args.lang

        if training_args.push_to_hub:
            trainer.push_to_hub(**kwargs)
        else:
            if output_path:
                trainer.save_model(output_path)

            trainer.create_model_card(**kwargs)


        return metrics

    
    results = {}

    if data_args.do_sft_train:
        # Supervised FineTuning (SFT) Training

        sft_data = load_data(data_args.sft_data_path)
        train_dataset = sft_data['train']
        eval_dataset = sft_data['test']

        if data_args.max_train_samples:
            subsample_dim = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(subsample_dim))
        
        if data_args.max_eval_samples:
            subsample_dim = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(subsample_dim))


        # Define optimizer
        num_update_steps_per_epoch = len(train_dataset)
        max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=max_train_steps
        )
        optimizers = (optimizer, lr_scheduler)

        run = wandb.init(
            project="UniBO-Chatbot", 
            name=f"SFT-FineTuning", 
            config={
                "architecture": "Decoder-only Transformer",
                "language": data_args.lang,
                "epochs": training_args.num_train_epochs,
            }
        )
        wandb.watch(model, log_freq=100)

        #response_template = " ### Answer:" if data_args.lang == 'en' else ' ### Risposta:'
        #collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

        # sft_train_dataset = ConstantLengthDataset(
        #     dataset=train_dataset, tokenizer=tokenizer
        # )

        # sft_eval_dataset = ConstantLengthDataset(
        #     dataset=eval_dataset, tokenizer=tokenizer
        # )

        sft_train_dataset = train_dataset
        sft_eval_dataset = eval_dataset
        print("ALLOCATED MEMORY",round(torch.cuda.memory_allocated(device) / 1000000000, 2), "GB")

        print(">>>>>",training_args.report_to)
        sft_trainer_config = {
            "model" : model,
            "tokenizer" : tokenizer,
            "train_dataset" : sft_train_dataset,
            "eval_dataset" : sft_eval_dataset,
            "args" : training_args,
            "max_seq_length" : data_args.max_seq_length,
            "formatting_func" : formatting_prompts_func,
            "optimizers" : optimizers,
            "compute_metrics" : compute_metrics
        }

        if model_args.use_peft:
            sft_trainer_config["peft_config"] = lora_config

        sft_trainer = SFTTrainer(
            **sft_trainer_config
        )

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint


        print("ALLOCATED MEMORY",round(torch.cuda.memory_allocated(device) / 1000000000, 2), "GB")
        sft_results = start_training(sft_trainer, checkpoint, train_type='sft')

        wandb.log(sft_results)
        wandb.finish()

        results['sft'] = sft_results

    if data_args.do_dpo_train:
        # Direct Policy Optimization (DPO) Training

        dpo_data = load_data(data_args.dpo_data_path)
        train_dataset = dpo_data['train']
        eval_dataset = dpo_data['test']

        if data_args.max_train_samples:
            subsample_dim = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(subsample_dim))
        
        if data_args.max_eval_samples:
            subsample_dim = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(subsample_dim))

        num_update_steps_per_epoch = len(train_dataset)
        max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=max_train_steps
        )
        optimizers = (optimizer, lr_scheduler)


        run = wandb.init(
            project="UniBO-Chatbot", 
            name=f"DPO-FineTuning", 
            config={
                "architecture": "Decoder-only Transformer",
                "language": data_args.lang,
                "epochs": training_args.num_train_epochs,
            }
        )
        wandb.watch(model, log_freq=100)

        # - load reference model
        model_ref_name = os.path.join(training_args.output_dir, f'_sft') if data_args.do_sft_train else model_args.model_name_or_path
        model_ref = AutoModelForCausalLM.from_pretrained(model_ref_name, cache_dir='cache/')


        dpo_trainer_config = {
            'model' : model,
            'model_ref' : model_ref,
            'args' : training_args,
            'beta' : 0.1,
            'train_dataset' : train_dataset, 
            'eval_dataset' : eval_dataset,
            'tokenizer' : tokenizer,
            'generate_during_eval' : False,
        }

        if model_args.use_peft:
            dpo_trainer_config["peft_config"] = lora_config

        dpo_trainer = DPOTrainer(
            **dpo_trainer_config
        )
        dpo_results = start_training(dpo_trainer, train_type='dpo')

        wandb.log(dpo_results)
        wandb.finish()

        results['dpo'] = dpo_results
    

    return results
    


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()