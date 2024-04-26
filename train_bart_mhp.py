import argparse
import copy
import torch
from typing import Optional, List, Union, Any, Dict
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from transformers import (
    HfArgumentParser,
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    Trainer,
    BertTokenizer,
    default_data_collator,
    is_torch_tpu_available,
    TrainingArguments,
    GPT2LMHeadModel,
    BartForConditionalGeneration,
    set_seed
)
from transformers.utils import check_min_version, send_example_telemetry
from data_utils.sp_tokenizer import SPTokenizer, BiTokenizer
import evaluate
import logging
import os 
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version
from transformers.trainer_utils import get_last_checkpoint
import transformers
from transformers.testing_utils import CaptureLogger
from itertools import chain
import math
import sys
import datasets
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from datasets import load_dataset
from collections import OrderedDict     
sys.path.append("/home/lsiyang/scratch/variable-text-segmentation")
from model.trainer import CustomizedTrainer

check_min_version("4.26.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    
    use_bi_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use different source and target tokenizers or not."},
    )
    
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    
    target_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Target tokenizer name or path if not the same as model_name"}
    )
    
    source_tokenizer_type: Optional[str] = field(
        default="wordpiece", metadata={"help": "three types: wordpiece, sentencepiece, bpe"}
    )    
    
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )
    
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_script: Optional[str] = field(
        default=None, metadata={"help": "The path of the script to load dataset."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
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

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    suffix: int = field(
        default=None, metadata={"help": "suffix number"}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None and self.dataset_script is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."
                
@dataclass 
class GenerationArguments:
    do_generation_test: Optional[bool] = field(
        default=False, metadata={"help": "generate on testset"})
    generate_length: Optional[int] = field(
        default=None, metadata={"help": "generation length"}
    )
    stop_token: Optional[str] = field(
        default=None, metadata={"help": "Token at which text generation is stopped"}
    )
    temperature: Optional[float] = field(
        default=None, metadata={"help": "Temperature of 1.0 has no effect, lower tend toward greedy sampling"}
    )  
    repetition_penalty: Optional[float] = field(
        default=1.0, metadata={"help": "Primarily useful for CTRL model; in that case, use 1.2"}
    ) 
    length_penalty: Optional[float] = field(
        default=0.0, metadata={"help": ">0.0 longer length"}
    ) 
    num_beams: Optional[int] = field(
        default=1, metadata={"help": "beam size"}
    )    
    exponential_decay_length_penalty: Optional[str] = field(
        default="(0,1)", metadata={"help": "exponential_decay_length_penalty"}
    ) 
    renormalize_logits: Optional[bool] = field(
        default=False, metadata={"help": "renormalize_logits"}
    ) 
    k: Optional[int] = field(
        default=50, metadata={"help": "top k"}
    )
    p: Optional[float] = field(
        default=0.9, metadata={"help": "top p"}
    )
    prefix: Optional[str] = field(
        default="", metadata={"help": "Text added prior to input."}
    )
    num_return_sequences: Optional[int] = field(
        default=1, metadata={"help": "The number of samples to generate."}
    )
@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    model_type: str = None
    pad_token_id: int = 0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if "bart" in self.model_type:
            batch = self.tokenizer.pad(
                [{"input_ids": itm["input_ids"], "attention_mask": itm["attention_mask"]} for itm in features],
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            decoder_batch = self.tokenizer.pad(
                [{"input_ids": itm["decoder_input_ids"], "attention_mask": itm["decoder_attention_mask"]} for itm in features],
                padding="longest",
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            batch["labels"] = copy.deepcopy(decoder_batch["input_ids"])
            # logger.info(f'labels{batch["labels"]}, decoder_input_id, {decoder_batch["input_ids"]}')
            # bart model wants to set padding to -100 so that it can be ignored when calculating the loss
            batch["labels"][batch["labels"] == self.pad_token_id] = -100
            # logger.info(f'labels{batch["labels"]}')
            batch["decoder_attention_mask"] = copy.deepcopy(decoder_batch["attention_mask"])
        else:
            batch = self.tokenizer.pad(
                features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            batch["labels"] = copy.deepcopy(batch["input_ids"])
        return batch    
    
@dataclass 
class TokenizerDataCollator():
    tokenizer: PreTrainedTokenizerBase
    model_args: ModelArguments
    dataCollatorWithPadding: DataCollatorWithPadding
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        examples = []
        for feature in features:
            examples.append(self.input_bart_function(feature))   
        logger.info(f'finish an epoch dataloader, example 1: {examples[0]["input_ids"]}')    
            
        return dataCollatorWithPadding(examples)
    
    def input_bart_function(self, example): 
        que_text ="[QUE] " + example["question"] + self.tokenizer.source_tokenizer.sep_token 
        desc_text = "[DESC] " + example["description"] + self.tokenizer.source_tokenizer.sep_token
        kwd_text =  "[KWD] " + example["topic"] + self.tokenizer.source_tokenizer.sep_token 
        ans_text = "[ANS] " + example["answer"] + self.tokenizer.source_tokenizer.sep_token + self.tokenizer.source_tokenizer.cls_token
        
        example["input_text"] = que_text + desc_text + kwd_text + ans_text
        example["ans_text"] = ans_text

            
        temp_res = {} 
        kwargs = {"add_special_tokens": False} 
        
        if  self.model_args.source_tokenizer_type == "sentencepiece":
            kwargs.update({"tokenize_with_sampling":training_args.do_train})    
        temp_res["que_token_res"] = self.tokenizer.source_tokenizer(que_text, **kwargs)
        temp_res["desc_token_res"] = self.tokenizer.source_tokenizer(desc_text, **kwargs)
        temp_res["kwd_token_res"] = self.tokenizer.source_tokenizer(kwd_text, **kwargs)
        if  self.model_args.model_type == "customed_bart":
            kwargs.update({"tokenize_with_sampling":training_args.do_train}) 
        temp_res["ans_token_res"] = self.tokenizer.target_tokenizer(ans_text, **kwargs)
        # temp_res["cls_token_res"] = self.tokenizer.target_tokenizer(self.tokenizer.target_tokenizer.cls_token, **kwargs)    
        # total_length.append(sum([len(temp_res[key]["input_ids"]) for key in temp_res]))  
        
        
        def fuc(cur):
            temp = temp_res["que_token_res"][cur] + temp_res["desc_token_res"][cur][:255]
            temp += temp_res["kwd_token_res"][cur]
            r1 = copy.deepcopy(temp)
            temp += temp_res["ans_token_res"][cur]
            r2 = temp[len(r1):512]
            return r1, r2
        

            
        (example["input_ids"], example["decoder_input_ids"]), (example["token_type_ids"],example["decoder_token_type_ids"]),\
            (example["attention_mask"], example["decoder_attention_mask"])= tuple(map(fuc, ["input_ids", "token_type_ids", "attention_mask"]))
        # logger.info(self.tokenizer.source_tokenizer.pad_token_id in example["decoder_input_ids"])
        assert len(example["input_ids"]) == len(example["token_type_ids"]) == len(example["attention_mask"]) 
        assert len(example["input_ids"]) <= 512
      
        return example
    


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, GenerationArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, generation_args = parser.parse_args_into_dataclasses()
    send_example_telemetry("run_clm", model_args, data_args)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    log_level = training_args.get_process_log_level()
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
    set_seed(training_args.seed)
    
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["valiadation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    elif data_args.dataset_script is not None:
        raw_datasets = load_dataset(
            data_args.dataset_script,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if generation_args.do_generation_test:
            del raw_datasets["train"]
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.valiadation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks    
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")
 
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    
    if model_args.tokenizer_name:
        if model_args.source_tokenizer_type == "wordpiece":
            if model_args.model_type=="bart":
                from transformers import BartTokenizer
                logger.info(f"According to tokenizer path{model_args.tokenizer_name}, load tokenizer from BartTokenizer class")
                tokenizer_kwargs.update({"model_input_names":["input_ids", "token_type_ids", "attention_mask"], "max_len": data_args.block_size})
                tokenizer = BartTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs) 
            else:    
                logger.info(f"According to tokenizer path{model_args.tokenizer_name}, load tokenizer from BertTokenizer class")
                tokenizer = BertTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)           
        elif model_args.source_tokenizer_type == "sentencepiece":
            logger.info(f"According to tokenizer path{model_args.target_tokenizer_name}, load tokenizer from Sentencepiece Tokenizer class")
            tokenizer_kwargs = {"name_or_path": model_args.target_tokenizer_name,
                "max_len": data_args.block_size,
                "model_input_names":["input_ids", "token_type_ids", "attention_mask"],
                "special_tokens_map_file": os.path.split(model_args.target_tokenizer_name)[0] + "/special_tokens_map.json",
                "align_pos": 50265,
                "model_max_length": data_args.block_size}   
            tokenizer = SPTokenizer(**tokenizer_kwargs)
        elif model_args.source_tokenizer_type == "bpe":
            if "bart" in model_args.model_type:
                from transformers import BartTokenizer
                logger.info(f"According to tokenizer path{model_args.tokenizer_name}, load tokenizer from BartTokenizer class")
                tokenizer_kwargs.update({"model_input_names":["input_ids", "token_type_ids", "attention_mask"], "max_len": data_args.block_size})
                tokenizer = BartTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs) 
            else:
                logger.info(f"According to tokenizer path{model_args.tokenizer_name}, load tokenizer from GPT2 Tokenizer class")
                tokenizer = GPT2Tokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)

    elif model_args.model_name_or_path:
        if model_args.source_tokenizer_type == "wordpiece":
            if model_args.model_type == "bart":
                from transformers import BartTokenizer
                logger.info(f"According to tokenizer path{model_args.model_name_or_path}, load tokenizer from BartTokenizer class")
                tokenizer_kwargs.update({"model_input_names":["input_ids", "token_type_ids", "attention_mask"], "max_len": data_args.block_size})
                tokenizer = BartTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs) 
            else:    
                logger.info(f"According to tokenizer path{model_args.model_name_or_path}, load tokenizer from BertTokenizer class")
                tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        elif model_args.source_tokenizer_type == "bpe":
            if model_args.model_type == "bart":
                from transformers import BartTokenizer
                logger.info(f"According to tokenizer path{model_args.tokenizer_name}, load tokenizer from BartTokenizer class")
                tokenizer_kwargs.update({"model_input_names":["input_ids", "token_type_ids", "attention_mask"], "max_len": data_args.block_size})
                tokenizer = BartTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs) 
            else:
                tokenizer = GPT2Tokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        elif model_args.source_tokenizer_type == "sentencepiece":
            tokenizer_kwargs = {"name_or_path": model_args.target_tokenizer_name,
                "max_len": data_args.block_size,
                "model_input_names":["input_ids", "token_type_ids", "attention_mask"],
                "special_tokens_map_file": os.path.split(model_args.target_tokenizer_name)[0] + "/special_tokens_map.json",
                "align_pos": 50265,
                "model_max_length": data_args.block_size}  
            tokenizer = SPTokenizer(**tokenizer_kwargs) 
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )   

    if model_args.use_bi_tokenizer:       
        if model_args.target_tokenizer_name: 
            model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
            # if "bart" in model_args.model_type:
            #     model_input_names = ["input_ids", "token_type_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask"]
            logger.info(f"According to target tokenizer path{model_args.target_tokenizer_name}, load tokenizer from Sentencepiece Tokenizer class")   
            tokenizer_kwargs = {"name_or_path": model_args.target_tokenizer_name,
                "max_len": data_args.block_size,
                "model_input_names":["input_ids", "token_type_ids", "attention_mask"],
                "special_tokens_map_file": os.path.split(model_args.target_tokenizer_name)[0] + "/special_tokens_map.json",
                "align_pos": 50265,
                "model_max_length": data_args.block_size}         
            target_tokenizer = SPTokenizer(**tokenizer_kwargs)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )             
        bi_tokenizer = BiTokenizer(tokenizer, target_tokenizer)
    else:
        bi_tokenizer = BiTokenizer(tokenizer)

    if model_args.model_name_or_path:
        if model_args.model_type == "customed":
            from model import customed_gpt2
            # and not (model_args.source_tokenizer_type == "sentencepiece")
            if training_args.do_train : 
                config = transformers.GPT2Config.from_json_file(model_args.config_name)
                model = customed_gpt2.CusVocab_GPT2LMHeadModel.from_pretrained(model_args.model_name_or_path, config=config, ignore_mismatched_sizes=True)
                load_model = transformers.GPT2LMHeadModel.from_pretrained(model_args.model_name_or_path)
                load_model_dict = load_model.state_dict()
                new_dict = model.state_dict()
                for key, parameter in new_dict.items():            
                    if key == "customed_lm_head.lm_head.weight":
                        print("load customed_lm_head.lm_head.weight")
                        new_dict[key][:load_model_dict["lm_head.weight"].size(0)] = load_model_dict["lm_head.weight"]
                    elif key == "customed_lm_head.lm_head.bias":
                        new_dict[key][:load_model_dict["lm_head.bias"].size(0)] = load_model_dict["lm_head.bias"]
                    elif key == "transformer.wte.weight":
                        print("load transformers.wte.weight")
                        new_dict[key][:load_model_dict["transformer.wte.weight"].size(0)] = load_model_dict["transformer.wte.weight"]
                    elif key in load_model_dict:
                        new_dict[key] = load_model_dict[key]
                        
                model.load_state_dict(new_dict)
            elif training_args.do_train and model_args.source_tokenizer_type == "sentencepiece":
                config = transformers.GPT2Config.from_json_file(model_args.config_name)
                model = customed_gpt2.CusVocab_GPT2LMHeadModel(config)
            else:
                model = customed_gpt2.CusVocab_GPT2LMHeadModel.from_pretrained(model_args.model_name_or_path)
                  
            if training_args.do_train: 
                model.customed_lm_head.get_mapping_weight(model.dtype, model.device) 
          
       
        elif model_args.model_type == "customed_bart":
            from model import customed_bart
            if training_args.do_train:
                
                config = transformers.BartConfig.from_json_file(model_args.config_name)
                model = customed_bart.CusVocab_BartLMHeadModel.from_pretrained(model_args.model_name_or_path, config=config, ignore_mismatched_sizes=True)   
                load_model = transformers.BartForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
                load_model_dict = load_model.state_dict()
                new_dict = model.state_dict()
                for key, parameter in new_dict.items():            
                    if key == "customed_lm_head.lm_head.weight":
                        print("load customed_lm_head.lm_head.weight")
                        print(new_dict[key][-1, :10])
                        new_dict[key][:load_model_dict["lm_head.weight"].size(0)] = load_model_dict["lm_head.weight"]
                    elif key == "customed_lm_head.lm_head.bias":
                        new_dict[key][:load_model_dict["lm_head.bias"].size(0)] = load_model_dict["lm_head.bias"]
                    # NOTE: double check later
                    elif key in {"model.shared.weight", "model.encoder.embed_tokens.weight", \
                        "model.decoder.embed_tokens.weight", "lm_head.weight"}:
                        print(f"load {key}")
                        print(new_dict[key][-1, :10])
                        new_dict[key][:load_model_dict[key].size(0)] = load_model_dict[key]
                    elif key == "final_logits_bias":
                        print("load final_logits_bias")
                        new_dict[key][:,:load_model_dict[key].size(1)] = load_model_dict[key]
                    elif key in load_model_dict:
                        new_dict[key] = load_model_dict[key]               
                model.load_state_dict(new_dict)                
            else:
                model = customed_bart.CusVocab_BartLMHeadModel.from_pretrained(model_args.model_name_or_path, ignore_mismatched_sizes=True)  
                load_model_dict = model.state_dict()
                for key, parameter in load_model_dict.items():            
                    if key == "customed_lm_head.lm_head.weight":
                        print("load customed_lm_head.lm_head.weight")
                        print(load_model_dict[key][-1, :10])
                    # NOTE: double check later
                    elif key in {"model.shared.weight", "model.encoder.embed_tokens.weight", \
                        "model.decoder.embed_tokens.weight", "lm_head.weight"}:
                        print(f"load {key}")
                        print(load_model_dict[key][-1,:10])
                    elif key == "customed_lm_head.mapping_lm_head.weight":
                        print("load customed_lm_head.mapping_lm_head.weight")
                        print(load_model_dict[key][-1, :10])
            if training_args.do_train: 
                model.customed_lm_head.init_lm_head_by_mapping(model.dtype, model.device)             
                load_model_dict = model.state_dict()
                for key, parameter in load_model_dict.items():            
                    if key == "customed_lm_head.lm_head.weight":
                        print("load customed_lm_head.lm_head.weight")
                        print(load_model_dict[key][-1, :10])
                    # NOTE: double check later
                    elif key in {"model.shared.weight", "model.encoder.embed_tokens.weight", \
                        "model.decoder.embed_tokens.weight", "lm_head.weight"}:
                        print(f"load {key}")
                        print(load_model_dict[key][-1,:10])
                    elif key == "customed_lm_head.mapping_lm_head.weight":
                        print("load customed_lm_head.mapping_lm_head.weight")
                        print(load_model_dict[key][-1, :10])
            n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
            logger.info(f"Loading pretrained model - Total size={n_params/2**20:.2f}M params")
        elif  model_args.model_type == "bart": 
            model = BartForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
            n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
            logger.info(f"Loading pretrained model - Total size={n_params/2**20:.2f}M params")
        else:
            model = GPT2LMHeadModel.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                ignore_mismatched_sizes=True
            )
            n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
            logger.info(f"Loading pretrained model - Total size={n_params/2**20:.2f}M params")
    else:
        raise RuntimeError("Need to specify model name or path!")
        model = GPT2LMHeadModel.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Loading new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    if model_args.use_bi_tokenizer:
        bi_tokenizer.source_tokenizer.add_tokens(["[QUE]", "[DESC]", "[KWD]", "[ANS]"], special_tokens=True)
        bi_tokenizer.target_tokenizer.add_tokens(["[QUE]", "[DESC]", "[KWD]", "[ANS]"], special_tokens=True)
    else:
        if data_args.dataset_config_name == "w strategy":
            bi_tokenizer.source_tokenizer.add_tokens(["[QUE]", "[DESC]", "[KWD]", "[ANS]", "[AR]", "[IN]", "[SELF]", "[DG]", "[OT]", "[RES]", "[INFO]"], special_tokens=True)
        else:
            bi_tokenizer.source_tokenizer.add_tokens(["[QUE]", "[DESC]", "[KWD]", "[ANS]"], special_tokens=True)
            logger.info(f'the ids of added tokens {["[QUE]", "[DESC]", "[KWD]", "[ANS]"]} are: {bi_tokenizer.source_tokenizer.encode(["[QUE]", "[DESC]", "[KWD]", "[ANS]"])}')
            logger.info(f'{bi_tokenizer.source_tokenizer.encode(["[QUE]"])} {bi_tokenizer.source_tokenizer.encode(["[DESC]"])} {bi_tokenizer.source_tokenizer.encode(["[KWD]"])} {bi_tokenizer.source_tokenizer.encode(["[ANS]"])}')
    if model_args.source_tokenizer_type == "bpe":
        added_tokens ={"sep_token": "<sep>", "pad_token": "<pad>", "cls_token": "<cls>", "mask_token": "<mask>", "unk_token": "<unk>"}  
        bi_tokenizer.source_tokenizer.add_special_tokens(added_tokens)
        logger.info(f"the ids of added tokens {list(added_tokens.values())} are: {bi_tokenizer.source_tokenizer.encode(list(added_tokens.values()))}")
    embedding_size = model.get_input_embeddings().weight.shape[0]
    logger.info(f"model input embeddings size: {embedding_size}")
    if len(bi_tokenizer.source_tokenizer) > embedding_size:
        model.resize_token_embeddings(len(bi_tokenizer.source_tokenizer))
    logger.info(f"model input embeddings size: {model.get_input_embeddings().weight.shape[0]}")       
        
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
        
    

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    total_length = []
    def input_function(example):
        with CaptureLogger(tok_logger) as cl:  
            que_text ="[QUE] " + example["question"].replace("\n","[SEP]") 
            desc_text = "[DESC] " + example["description"].replace("\n","[SEP]")
            kwd_text =  "[KWD] " + example["keywords"].replace("\n","[SEP]") 
            ans_text = "[ANS] " + example["answer"].replace("\n","[SEP]") 
            
            example["input_text"] = que_text + desc_text + kwd_text + ans_text
            example["ans_text"] = ans_text

               
            temp_res = {} 
            kwargs = {"add_special_tokens": False} 
            if  model_args.source_tokenizer_type == "sentencepiece":
                kwargs.update({"tokenize_with_sampling":training_args.do_train})    
            temp_res["que_token_res"] = bi_tokenizer.source_tokenizer(que_text, **kwargs)
            temp_res["desc_token_res"] = bi_tokenizer.source_tokenizer(desc_text, **kwargs)
            temp_res["kwd_token_res"] = bi_tokenizer.source_tokenizer(kwd_text, **kwargs)
            if  model_args.model_type == "customed":
                kwargs.update({"tokenize_with_sampling":training_args.do_train}) 
            temp_res["ans_token_res"] = bi_tokenizer.target_tokenizer(ans_text, **kwargs)
            temp_res["cls_token_res"] = bi_tokenizer.target_tokenizer("[CLS]", **kwargs)             
            total_length.append(sum([len(temp_res[key]["input_ids"]) for key in temp_res]))
            
             
        
            
            def fuc(cur):
                temp = temp_res["que_token_res"][cur] + temp_res["desc_token_res"][cur][:511]
                temp += temp_res["kwd_token_res"][cur] + temp_res["ans_token_res"][cur]
                temp = temp[:1023]
                temp += temp_res["cls_token_res"][cur]
                return temp 
            


            example["input_ids"], example["token_type_ids"], example["attention_mask"] = tuple(map(fuc, ["input_ids", "token_type_ids", "attention_mask"]))
            ans_pos = example["input_ids"].index(bi_tokenizer.target_tokenizer.convert_tokens_to_ids("[ANS]"))
            example["label_pos_ids"] = [0] * ans_pos + [1] * (len(example["input_ids"]) - ans_pos)
            assert len(example["input_ids"]) == len(example["token_type_ids"]) == len(example["attention_mask"]) == len(example["label_pos_ids"])
            assert len(example["input_ids"]) <= 1024
      
        return example

    def input_bart_function(example):
        with CaptureLogger(tok_logger) as cl:  
            que_text ="[QUE] " + example["question"] + bi_tokenizer.source_tokenizer.sep_token 
            desc_text = "[DESC] " + example["description"] + bi_tokenizer.source_tokenizer.sep_token
            kwd_text =  "[KWD] " + example["topic"] + bi_tokenizer.source_tokenizer.sep_token 
            ans_text = "[ANS] " + example["answer"] + bi_tokenizer.source_tokenizer.sep_token + bi_tokenizer.source_tokenizer.cls_token
            
            example["input_text"] = que_text + desc_text + kwd_text + ans_text
            example["ans_text"] = ans_text

               
            temp_res = {} 
            kwargs = {"add_special_tokens": False} 
            
            if  model_args.source_tokenizer_type == "sentencepiece":
                kwargs.update({"tokenize_with_sampling":training_args.do_train})    
            temp_res["que_token_res"] = bi_tokenizer.source_tokenizer(que_text, **kwargs)
            temp_res["desc_token_res"] = bi_tokenizer.source_tokenizer(desc_text, **kwargs)
            temp_res["kwd_token_res"] = bi_tokenizer.source_tokenizer(kwd_text, **kwargs)
            if  model_args.model_type == "customed_bart":
                kwargs.update({"tokenize_with_sampling":training_args.do_train}) 
            temp_res["ans_token_res"] = bi_tokenizer.target_tokenizer(ans_text, **kwargs)
            # temp_res["cls_token_res"] = bi_tokenizer.target_tokenizer(bi_tokenizer.target_tokenizer.cls_token, **kwargs)    
            total_length.append(sum([len(temp_res[key]["input_ids"]) for key in temp_res]))  
            
            
            def fuc(cur):
                temp = temp_res["que_token_res"][cur] + temp_res["desc_token_res"][cur][:255]
                temp += temp_res["kwd_token_res"][cur]
                r1 = copy.deepcopy(temp)
                temp += temp_res["ans_token_res"][cur]
                r2 = temp[len(r1):512]
                return r1, r2
            

                
            (example["input_ids"], example["decoder_input_ids"]), (example["token_type_ids"],example["decoder_token_type_ids"]),\
                (example["attention_mask"], example["decoder_attention_mask"])= tuple(map(fuc, ["input_ids", "token_type_ids", "attention_mask"]))
            # logger.info(bi_tokenizer.source_tokenizer.pad_token_id in example["decoder_input_ids"])
            assert len(example["input_ids"]) == len(example["token_type_ids"]) == len(example["attention_mask"]) 
            assert len(example["input_ids"]) <= 512
      
        return example
    torch.set_printoptions(profile="full")
    
    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:            
            output = tokenizer(examples["input"])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output
    
    with training_args.main_process_first(desc="dataset map tokenization"):
        if "bart" in model_args.model_type:
            lm_datasets = raw_datasets.map(
                input_bart_function,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running posprocess on dataset",
            )
        else:
            lm_datasets = raw_datasets.map(
                input_function,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running posprocess on dataset",
            )
    
    # lm_datasets = raw_datasets
        

        
    if data_args.block_size is None:
        block_size = bi_tokenizer.source_tokenizer.model_max_length
        if block_size > 512:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({bi_tokenizer.source_tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 512
    else:
        if data_args.block_size > bi_tokenizer.source_tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({bi_tokenizer.source_tokenizer.model_max_length}). Using block_size={bi_tokenizer.source_tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, bi_tokenizer.source_tokenizer.model_max_length)
    
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"]
        

        return result  
    
   
    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))


    if training_args.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")
        

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            # labels = labels[:, 1:].reshape(-1)
            labels = labels[:, :-1].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            logger.info(f"labels, {labels}, preds, {preds}, {(labels[labels!=-100]).shape}")
            # labels_new = labels[labels!=-100]
            # preds = preds[labels!=-100]
            # labels = labels_new
            return metric.compute(predictions=preds, references=labels)
        
    data_collator_padding = DataCollatorWithPadding(max_length=512, tokenizer=bi_tokenizer.target_tokenizer, \
        padding="max_length", model_type=model_args.model_type, pad_token_id = bi_tokenizer.target_tokenizer.pad_token_id)
    data_collator_tokenizer = TokenizerDataCollator(tokenizer=bi_tokenizer, model_args=model_args, dataCollatorWithPadding=data_collator_padding)
    logger.info(f"bi_tokenizer.target_tokenizer.pad_token_id: {bi_tokenizer.target_tokenizer.pad_token_id}")
    if training_args.do_train or training_args.do_eval:
        # Initialize our Trainer
        trainer = CustomizedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=bi_tokenizer.target_tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            data_collator=data_collator_padding,
            compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None,
        )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    



        
    def adjust_length_to_model(length, max_sequence_length):
        if length < 0 and max_sequence_length > 0:
            length = max_sequence_length
        elif 0 < max_sequence_length < length:
            length = max_sequence_length  # No generation bigger than model size
        elif length < 0:
            length = MAX_LENGTH  # avoid infinite loop
        return length
        
    if generation_args.do_generation_test:
        if "test" not in lm_datasets:
            raise ValueError("--do_generation_test requires a test dataset")
        test_dataset = lm_datasets["test"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        model.to(device)
        
        logger.info("*** Test Generation ***")
        logger.info("wrap model")
        
        test_data_loader= DataLoader(
                test_dataset,
                batch_size=1,
        )
        results = []
        predictions = []
        references = []
        pred_dict = OrderedDict()
        ref_dict = OrderedDict()
        counter = [0, 0]
        if generation_args.exponential_decay_length_penalty:
            generation_args.exponential_decay_length_penalty = tuple(eval(generation_args.exponential_decay_length_penalty))
        print(generation_args)
        for inputs in test_data_loader:
            prompt_text = inputs["input_text"][0].split("[ANS]")[0]
            oracle_text = inputs["input_text"][0].split("[ANS]")[1].replace(bi_tokenizer.target_tokenizer.cls_token, "").replace(bi_tokenizer.target_tokenizer.sep_token, " ")
            # print(prompt_text)
            encoded_prompt_ids = bi_tokenizer.source_tokenizer.encode(
                prompt_text, add_special_tokens=False
            )
            encoded_prompt = [encoded_prompt_ids[:512]]
            encoded_prompt = torch.tensor(encoded_prompt).to(torch.long)
            prompt = torch.tensor([[model.config.decoder_start_token_id]]).to(device).to(torch.long)
            encoded_prompt = encoded_prompt.to(device)
            if encoded_prompt.size()[-1] == 0:
                input_ids = None
            else:
                input_ids = encoded_prompt
            encoder_outputs = model.get_encoder()(input_ids)
            import time
            start = time.time()  
            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=generation_args.generate_length,
                min_length= 5,
                temperature=generation_args.temperature,
                top_k=generation_args.k,
                top_p=generation_args.p,
                repetition_penalty=generation_args.repetition_penalty,
                do_sample=True,
                early_stopping=True,
                num_beams=generation_args.num_beams,
                renormalize_logits=generation_args.renormalize_logits,
                length_penalty=generation_args.length_penalty,
                exponential_decay_length_penalty=generation_args.exponential_decay_length_penalty,
                num_return_sequences=generation_args.num_return_sequences,
                pad_token_id=bi_tokenizer.target_tokenizer.pad_token_id,
                eos_token_id=bi_tokenizer.target_tokenizer.cls_token_id,
                use_cache=True
                
            )
            # input_ids=prompt,
            # encoder_outputs = encoder_outputs
            
            infer_time = time.time()-start
            counter[0] = counter[0] + 1
            counter[1] = counter[1] + infer_time
            # num_beams=5,
            if counter[0] % 100 == 0:
                logger.info(f"count: {counter[0]}, avg seconds:{counter[1]/counter[0]}")
            # Remove the batch dimension when returning multiple sequences
            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()


            for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                logger.info(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
                generated_sequence = generated_sequence.tolist()
                
                # Decode text
                text = bi_tokenizer.target_tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
                
                # # Remove all text after the stop token
                text = text[text.find("[ANS]")+5: text.find(bi_tokenizer.target_tokenizer.cls_token)]
                text_tokens = bi_tokenizer.target_tokenizer.convert_ids_to_tokens(generated_sequence)
                text_tokens = text_tokens[text_tokens.index("[ANS]")+1 : ]
                if bi_tokenizer.target_tokenizer.cls_token in text_tokens:
                    text_tokens = text_tokens[:text_tokens.index(bi_tokenizer.target_tokenizer.cls_token)-1]
                generated_response = text.replace(bi_tokenizer.target_tokenizer.cls_token, "").replace(bi_tokenizer.target_tokenizer.sep_token, " ").strip()           
                results.append({"context": prompt_text, "generated": generated_response, "tokens": text_tokens, "oracle": oracle_text})
                # logger.info(generated_response)
                if prompt_text in ref_dict:
                    
                    pred_dict[prompt_text].append(" ".join(generated_response.split(" ")))
                    ref_dict[prompt_text].add(" ".join(oracle_text.split(" ")))
                else:
                    pred_dict[prompt_text] = []
                    ref_dict[prompt_text] = set()
                    
                    pred_dict[prompt_text].append(" ".join(generated_response.split(" ")))
                    ref_dict[prompt_text].add(" ".join(oracle_text.split(" ")))
                    
                
                    
        predictions = []
        references = []                              
        for key, values in pred_dict.items():
            reference = list(ref_dict[key])
            for value in values:
                predictions.append(value)
                references.append(reference)
                 
        print(predictions)      
        bleu = evaluate.load("bleu")
        bleu_res = bleu.compute(predictions=predictions, references=references)
        print(bleu_res)
        rouge = evaluate.load('rouge')
        rouge_res = rouge.compute(predictions=predictions, references=references)
        print(rouge_res)
        bertscore = evaluate.load("bertscore")
        bertscore_res = bertscore.compute(predictions=predictions, references=references, lang="en")
        print("bertscore_res:", [sum(bertscore_res[col])/len(bertscore_res[col]) for col in bertscore_res if col != "hashcode"] )
        import json
        suffix = (training_args.output_dir).split("/")[-1] if (training_args.output_dir).split("/")[-1] is not "" else (training_args.output_dir).split("/")[-2]
        # suffix = data_args.suffix
        json.dump(results, open(f"evaluate_{suffix}_k{generation_args.k}_t{generation_args.temperature}_p{generation_args.p}_r{generation_args.repetition_penalty}_exp{generation_args.exponential_decay_length_penalty[0]}_{generation_args.exponential_decay_length_penalty[1]*100}_beam_wooutput.out", "w"), indent=2, ensure_ascii=False)
           
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()
    
if __name__ == "__main__":
    main()   

    