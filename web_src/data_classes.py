from dataclasses import dataclass, field
from typing import (
    Optional,
    List
)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    max_seq_length: Optional[int] = field(
        default=4096,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    
    # Training type 
    do_sft_train : Optional[bool] = field(
        default = False,
        metadata = {
            "help" : "Whether to run Supervised-FineTuning (SFT)."
        }
    )

    do_dpo_train : Optional[bool] = field(
        default = False,
        metadata = {
            "help" : "Whether to run Direct Preference Optimization (DPO)."
        }
    )    

    # Data repo
    sft_data_path : Optional[str] = field(
        default='data/faqs/sft_data.json',
        metadata={
            "help" : "Path to data for the SFT finetuning."
        }
    )

    dpo_data_path : Optional[str] = field(
        default='data/faqs/dpo_data.json',
        metadata={
            "help" : "Path to data for the DPO finetuning."
        }
    )

    data_from_dict : Optional[bool] = field(
        default = False,
        metadata = {
            'help' : 'Whether data is loaded from a python dictionary.'
        }
    )

    split_data : Optional[bool] = field(
        default = False,
        metadata = {
            "help" : "Whether to split data into {train, evaluation}"
        }
    )

    train_test_split_ratio : Optional[float] = field(
        default = None,
        metadata = {
            "help" : "Split ratio to use for the the test."
        }
    )

    lang : Optional[str] = field(
        default='it',
        metadata={
            "help" : "Language used in the input FAQs ('it' or 'en')."
        }
    )

    # Retrieval data
    data_repository_path : Optional[str] = field(
        default='data/',
        metadata={
            "help" : "Path to reference data for retrieval."
        }
    )

    retrieval_repo_is_directory : Optional[bool] = field(
        default=False,
        metadata= {
            "help" : "Whether the data to retrieve facts from is a directory of separate files."
        }
    )

    num_false_options_to_generate : Optional[int] = field(
        default=3,
        metadata={
            "help" : "Number of false options for each question to generate."
        }
    )

    max_num_chunk_for_generation : Optional[int] = field(
        default=None,
        metadata = {
            "help" : "Number of chunks to use during the RAG inference phase."
        }
    )

   

    # Other configs
    overwrite_cache:  Optional[bool] = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length:  Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    source_prefix :  Optional[str] = field(
        default='',
        metadata={"help" : "Prefix to put in front of the input data istances."}
    )

    preprocessing_num_workers : Optional[int] = field(
        default=None,
        metadata={'help' : 'Degree of parallelism specifying the number of worker processes to use for preprocessing.'}
    )

    push_hub : Optional[bool] = field(
        default=False, 
        metadata={"help":"Whether or not to push the model to the Hub."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_name_local: Optional[str] = field(
        default=None, metadata={"help": "The name of the local dataset to use."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    evaluator_model_name_or_path : Optional[str] = field(
        default=None,
        metadata={"help" : "Name of the model used to evaluate the generated samples."}
    )

    embedder_model_name : Optional[str] = field(
        default = 'local',
        metadata = {"help" : "Embedder model's name to use for encoding and chunk retrieval."}
    )

    model_for_bertscore: Optional[str] = field(
        default="distilbert-base-uncased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models for bertscore"}
    )

    use_retrieval : Optional[bool] = field(
        default = False,
        metadata = {"help" : "Whether to use retrieal to inject knowledge into the inference pipeline."}
    )

    # Model context config
    context_window : Optional[int] = field(
        default = 2048,
        metadata = {"help" : "Context window dimension."} 
    )

    max_new_tokens : Optional[int] = field(
        default = 512,
        metadata = {"help" : "Maximum number of tokens the model is allowed to generate."}
    )

    # Retriever config
    chunk_size : Optional[int] = field(
        default = 512,
        metadata = {"help" : ""}
    )

    chunk_overlap : Optional[float] = field(
        default = 0.0,
        metadata = {"help" : "Overlapping percentage betweem chunks from the reference data source."}
    )

    top_k_similarity : Optional[int] = field(
        default = 2,
        metadata = {"help" : "Number of chunks to retrieve for each query based on similarity scores."}
    )

    # Others
    use_peft: Optional[bool] = field(
        default=False,
        metadata={"help": "Use PEFT for training or not"},
    )
    use_vllm_engine : Optional[bool] = field(
        default = False,
        metadata = {"help" : "Whether to use models from the vLLM library for optimized inference."}
    )
    quantize_4_bits : Optional[bool] = field(
        default = False,
        metadata = {"help" : "Whether to quantize the model with 4-bit precision."}
    )
    config_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: Optional[bool] = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

    ## Sampling parameters
    do_sample : Optional[bool] = field(
        default = False,
        metadata = {"help" : "Enable decoding sampling techniques."}
    )

    temperature : Optional[float] = field(
        default=0.0,
        metadata={"help" : "Sampling temperature for the text generation."}
    )

    top_p : Optional[float] = field(
        default = 1.0,
        metadata={
            "help" : "If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation."}
    )

    top_k : Optional[int] = field(
        default=50,
        metadata={"help":"The number of highest probability vocabulary tokens to keep for top-k-filtering."}
    )
