import json

import sentencepiece as spm
import torch.nn
from datasets import load_metric
from transformers import BartTokenizer, EvalPrediction, BartForQuestionAnswering, AutoConfig, \
    BartForConditionalGeneration

from adapted_model import CusVocab_BartQAModel
from sp_tokenizer import SPTokenizer
from utils_qa import postprocess_qa_predictions


class Mapping(torch.nn.Module):
    def __init__(self, mapping_file):
        super(Mapping, self).__init__()
        with open(mapping_file, "r", encoding="utf-8") as f:
            self.mapping = json.load(f)

        def forward(self, input_ids):
            return input_ids.map(lambda x: self.mapping[x])


# train a sentencepiece model on the SQUAD dataset
def train_sentencepiece_model_on_squad():
    # Load the SQUAD dataset through the Hugging Face datasets library
    from datasets import load_dataset
    squad_dataset = load_dataset("squad")

    # Extract the context and question texts from the dataset
    contexts = squad_dataset['train']['context']
    questions = squad_dataset['train']['question']

    # Combine the context and question texts into a single list
    combined_texts = [f"{context} {question}" for context, question in zip(contexts, questions)]

    # Write the combined texts to a text file
    with open("squad_combined_texts.txt", "w", encoding="utf-8") as f:
        for text in combined_texts:
            f.write(text + "\n")

    # Train a sentencepiece model on the combined texts
    spm.SentencePieceTrainer.train(input="squad_combined_texts.txt", model_prefix="./new_tokenizer/squad_spm",
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


# %%
def train_model():  # For BART LM Head (need to be adapted for question answering)
    # load base tokenizer
    base_tokenizer = BartTokenizer.from_pretrained("./base_tokenizer")
    # load tokenizer
    # Initialize the SP tokenizer
    kwargs = {
        "name_or_path": "./new_tokenizer/squad_spm.model",
        "max_len": 1024,
        "model_input_names": ["input_ids", "token_type_ids", "attention_mask"],
        "special_tokens_map_file": "./new_tokenizer/special_tokens_map.json",
        "align_pos": 50265,
    }
    tokenizer = SPTokenizer(**kwargs)

    # load model config
    config = AutoConfig.from_pretrained("facebook/bart-base")
    # add base_vocab_size
    config.__setattr__("base_vocab_size", 50265)
    # add vocab_size
    config.__setattr__("vocab_size", base_tokenizer.vocab_size)
    config.__setattr__("lazy_mapping_weight", False)
    config.__setattr__("mapping_index_file", "./new_tokenizer/mapping.json")

    model = BartForQuestionAnswering.from_pretrained("facebook/bart-base")

    new_tokens = tokenizer.vocab_size - base_tokenizer.vocab_size
    embedding = model.resize_token_embeddings(new_tokens)

    def preprocess_training_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = base_tokenizer(
            questions,
            examples["context"],
            max_length=1024,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=False,
        )

        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for answer in answers:
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])

            # append start and end position
            start_positions.append(start_char)
            end_positions.append(end_char)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def preprocess_validation_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = base_tokenizer(
            questions,
            examples["context"],
            max_length=1024,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=False,
            padding="max_length",
        )

        example_ids = []

        for i in range(len(inputs["input_ids"])):
            example_ids.append(examples["id"][i])

        inputs["example_id"] = example_ids
        return inputs

    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=False,
            n_best_size=20,
            max_answer_length=30,
            null_score_diff_threshold=0.0,
            output_dir=training_args.output_dir,
            log_level=3,
            prefix=stage,
        )
        formatted_predictions = [{"id": str(k), "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": str(ex["id"]), "answers": ex["answers"]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Load the SQUAD dataset through the Hugging Face datasets library
    from datasets import load_dataset
    dataset = load_dataset("squad")

    tokenized_train_dataset = dataset["train"].map(preprocess_training_examples, batched=True,
                                                   remove_columns=dataset["train"].column_names)
    tokenized_validation_dataset = dataset["validation"].map(preprocess_validation_examples, batched=True,
                                                             remove_columns=dataset["validation"].column_names)

    # Load trainer
    from transformers import TrainingArguments
    from transformers import Trainer
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        fp16=True,
        per_device_train_batch_size=6,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        tokenizer=base_tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # post processing with post_processing_function
    # trainer.evaluate(eval_dataset=tokenized_validation_dataset, post_process_function=post_processing_function)


# %%
if __name__ == "__main__":
    print("Training sentencepiece model on SQUAD dataset")
    # %%
    # train_sentencepiece_model_on_squad()
    # %% reindex
    reindex_tokenizer_to_sentencepiece_model()
    # %%
    print("Mapping tokenizer to sentencepiece model")
    map_tokenizer_to_sentencepiece_model()
    # %%
    print("Training model")
    train_model()
