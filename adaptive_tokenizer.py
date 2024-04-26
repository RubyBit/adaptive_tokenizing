import json
import os
from typing import Optional

import numpy as np
import safetensors
import sentencepiece as spm
import torch.nn
from datasets import load_metric
from transformers import BartTokenizer, EvalPrediction, AutoConfig, \
    BartForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, PreTrainedModel
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.utils import WEIGHTS_NAME, SAFE_WEIGHTS_NAME

from adapted_model import CusVocab_BartLMHeadModel, logger
from sp_tokenizer import SPTokenizer


class CustomTrainer(Seq2SeqTrainer):

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        ##if self.tokenizer is not None:
        ##    self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

# train a sentencepiece model on the SQUAD dataset
def train_sentencepiece_model_on_billsum():
    # Load the SQUAD dataset through the Hugging Face datasets library
    from datasets import load_dataset
    squad_dataset = load_dataset("billsum")

    # Extract the context and question texts from the dataset
    text = squad_dataset['train']['text']
    summary = squad_dataset['train']['summary']

    # Combine the context and question texts into a single list
    combined_texts = [f"{text}, {summary}" for text, summary in zip(text, summary)]

    # Write the combined texts to a text file
    with open("billsum_combined.txt", "w", encoding="utf-8") as f:
        for text in combined_texts:
            f.write(text + "\n")

    # Train a sentencepiece model on the combined texts
    spm.SentencePieceTrainer.train(input="billsum_combined.txt", model_prefix="./new_tokenizer/squad_spm",
                                   model_type="unigram", vocab_size=6000, split_by_whitespace=False,
                                   max_sentencepiece_length=32, remove_extra_whitespaces=True, add_dummy_prefix=False)


# %%
def map_tokenizer_to_sentencepiece_model():
    # Initialize the BART tokenizer
    bart_tokenizer = BartTokenizer.from_pretrained("./base_tokenizer")
    bart_tokenizer.add_special_tokens({"sep_token": "<sep>"})
    bart_tokenizer.add_special_tokens({"cls_token": "<cls>"})
    bart_tokenizer.add_special_tokens({"pad_token": "<pad>"})
    bart_tokenizer.add_special_tokens({"additional_special_tokens": ["[QUE]", "[DESC]", "[KWD]", "[ANS]"]})

    # Initialize the SP tokenizer
    kwargs = {
        "name_or_path": "./new_tokenizer/squad_spm.model",
        "max_len": 1024,
        "model_input_names": ["input_ids", "token_type_ids", "attention_mask"],
        "special_tokens_map_file": "./new_tokenizer/special_tokens_map.json",
        "align_pos": 50265,
    }
    sp_tokenizer = SPTokenizer(**kwargs)

    sp_tokenizer.build_mapping_file(bart_tokenizer, "./new_tokenizer/mapping.json")


def reindex_tokenizer_to_sentencepiece_model():
    # load bart tokenizer
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    # save to "/base_tokenizer"
    bart_tokenizer.save_pretrained("./base_tokenizer")

    kwargs = {"name_or_path": "./new_tokenizer/squad_spm.model",
              "max_len": 1024,
              "model_input_names": ["input_ids", "token_type_ids", "attention_mask"],
              "special_tokens_map_file": "./new_tokenizer/special_tokens_map.json",
              "align_pos": 50265}
    tokenizer = SPTokenizer(**kwargs)
    control_tokens = {"sep_token": "<sep>", "pad_token": "<pad>", "cls_token": "<cls>", "mask_token": "<mask>"}
    unknown_token = {"unk_token": "<unk>"}

    tokenizer.reindex_according_wordpiece("./base_tokenizer/vocab.json", output_dir="./new_tokenizer",
                                          control_tokens=control_tokens, unknown_token=unknown_token,
                                          is_chinese_vocab=False, whitespace_placeholder="Ġ", base_score=-100)


def test_decode_sp_tokenizer():
    # load tokenizer
    kwargs = {
        "name_or_path": "./new_tokenizer/squad_spm.model",
        "max_len": 1024,
        "model_input_names": ["input_ids", "token_type_ids", "attention_mask"],
        "special_tokens_map_file": "./new_tokenizer/special_tokens_map.json",
        "align_pos": 50265,
    }
    tokenizer = SPTokenizer(**kwargs)
    config = AutoConfig.from_pretrained("facebook/bart-base")
    # add base_vocab_size
    config.__setattr__("base_vocab_size", 50265)
    # add vocab_size
    config.__setattr__("vocab_size", tokenizer.vocab_size)
    config.__setattr__("lazy_mapping_weight", False)
    config.__setattr__("mapping_index_file", "./new_tokenizer/mapping.json")
    # load model
    model = CusVocab_BartLMHeadModel(config)

    # load dataset
    from datasets import load_dataset
    dataset = load_dataset("billsum")

    # Tokenize the examples
    inputs = dataset["train"]["text"][:2]
    inputs = [f"summarize: {input}" for input in inputs]
    inputs = tokenizer(inputs, max_length=1024, padding="max_length", truncation=True, return_tensors="pt")

    # print type of inputs
    print(f"type of inputs: {type(inputs['input_ids'])}")
    # convert to np.array
    inputs_ids = inputs["input_ids"].numpy()
    # convert to torch.tensor
    inputs_ids = torch.tensor(inputs_ids)
    # Decode the tokenized inputs
    decoded_inputs = tokenizer.batch_decode(inputs_ids, skip_special_tokens=True)
    print(decoded_inputs)

    # Generate the outputs
    outputs = model.generate(inputs["input_ids"], max_length=1024, num_beams=4, early_stopping=True)
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(decoded_outputs)


# %%
def train_model():  # For BART LM Head (need to be adapted for question answering)
    # load base tokenizer
    old_tokenizer = BartTokenizer.from_pretrained("./base_tokenizer")
    # load tokenizer
    # Initialize the SP tokenizer
    kwargs = {
        "name_or_path": "./new_tokenizer/squad_spm.model",
        "max_len": 1024,
        "model_input_names": ["input_ids", "token_type_ids", "attention_mask"],
        "special_tokens_map_file": "./new_tokenizer/special_tokens_map.json",
        "align_pos": 50265,
    }
    base_tokenizer = SPTokenizer(**kwargs)

    # load model config
    config = AutoConfig.from_pretrained("facebook/bart-base")
    # add base_vocab_size
    config.__setattr__("base_vocab_size", 50265)
    # add vocab_size
    config.__setattr__("vocab_size", base_tokenizer.vocab_size)
    config.__setattr__("lazy_mapping_weight", False)
    config.__setattr__("mapping_index_file", "./new_tokenizer/mapping.json")
    # load mapping_index
    with open("./new_tokenizer/mapping.json", "r") as f:
        mapping_index = json.load(f)
        mapping_index_matrix = [[0 if x is None else x for x in row] for row in mapping_index]
        print(mapping_index_matrix[:10])

    model = CusVocab_BartLMHeadModel(config)
    # load pretrained model
    load_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    load_model_dict = load_model.state_dict()
    new_dict = model.state_dict()
    print(f"load_model_dict keys: {load_model_dict.keys()}")
    print(f"new_dict keys: {new_dict.keys()}")

    for key, parameter in new_dict.items():
        if key == "customed_lm_head.lm_head.weight":
            print("load customed_lm_head.lm_head.weight")
            print(new_dict[key][-1, :10])
            new_dict[key][:load_model_dict["lm_head.weight"].size(0)] = load_model_dict["lm_head.weight"]
        elif key == "customed_lm_head.lm_head.bias":
            new_dict[key][:load_model_dict["lm_head.bias"].size(0)] = load_model_dict["lm_head.bias"]
        # NOTE: double check later
        elif key in {"model.shared.weight", "model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight",
                     "lm_head.weight"}:
            print(f"load {key}")
            print(new_dict[key][-1, :10])
            new_dict[key][:load_model_dict[key].size(0)] = load_model_dict[key]
        elif key == "final_logits_bias":
            print("load final_logits_bias")
            new_dict[key][:, :load_model_dict[key].size(1)] = load_model_dict[key]
        elif key in load_model_dict:
            new_dict[key] = load_model_dict[key]

    model.load_state_dict(new_dict)

    model.customed_lm_head.init_lm_head_by_mapping(model.dtype, model.device)

    load_model_dict = model.state_dict()
    for key, parameter in load_model_dict.items():
        if key == "customed_lm_head.lm_head.weight":
            print("load customed_lm_head.lm_head.weight")
            print(load_model_dict[key][-1, :10])
        # NOTE: double check later
        elif key in {"model.shared.weight", "model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight",
                     "lm_head.weight"}:
            print(f"load {key}")
            print(load_model_dict[key][-1, :10])
        elif key == "customed_lm_head.mapping_lm_head.weight":
            print("load customed_lm_head.mapping_lm_head.weight")
            print(load_model_dict[key][-1, :10])
    prefix = "summarize: "

    def preprocess_training_examples(examples):
        # Tokenize the examples
        inputs = [prefix + example for example in examples["text"]]
        model_inputs = base_tokenizer(inputs, max_length=1024, padding="max_length", truncation=True,
                                      return_tensors="pt")

        labels = base_tokenizer(examples["summary"], max_length=1024, padding="max_length", truncation=True,
                                return_tensors="pt")

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_validation_examples(examples):
        # Tokenize the examples
        inputs = [prefix + example for example in examples["text"]]
        model_inputs = base_tokenizer(inputs, max_length=1024, padding="max_length", truncation=True,
                                      return_tensors="pt")

        labels = base_tokenizer(examples["summary"], max_length=1024, padding="max_length", truncation=True,
                                return_tensors="pt")

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    metric = load_metric("rouge")

    def compute_metrics(p: EvalPrediction):
        predictions, labels = p
        # print type of predictions and labels
        print(f"type of predictions: {type(predictions)}")
        print(f"type of labels: {type(labels)}")
        predictions = base_tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, base_tokenizer.pad_token_id)
        decoded_labels = base_tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = metric.compute(predictions=predictions, references=decoded_labels, use_stemmer=True)

        # Extract a few results
        prediction_lens = [np.count_nonzero(pred != base_tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}
        return result

    data_collator = DataCollatorForSeq2Seq(base_tokenizer, model=model)

    # Load the SQUAD dataset through the Hugging Face datasets library
    from datasets import load_dataset
    dataset = load_dataset("billsum")

    tokenized_train_dataset = dataset["train"].map(preprocess_training_examples, batched=True,
                                                   remove_columns=dataset["train"].column_names)
    tokenized_validation_dataset = dataset["test"].map(preprocess_validation_examples, batched=True,
                                                       remove_columns=dataset["test"].column_names)

    # Load trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        fp16=True,
        per_device_train_batch_size=6,
        predict_with_generate=True,
    )
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        tokenizer=base_tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    trainer.train()

    # post processing with post_processing_function
    # trainer.evaluate(eval_dataset=tokenized_validation_dataset, post_process_function=post_processing_function)


# %%
if __name__ == "__main__":
    train_model()
