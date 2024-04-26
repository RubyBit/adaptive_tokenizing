import collections

from transformers import AutoTokenizer, LukeForQuestionAnswering, Trainer, TrainingArguments, AutoModel, \
    BertForQuestionAnswering
from datasets import load_dataset, load_metric
import torch
import json

from tokenizers import Tokenizer, trainers, pre_tokenizers, decoders, models, normalizers

# wordpiece tokenizer
tokenizer = Tokenizer(models.Unigram())

tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
tokenizer.decoder = decoders.WordPiece()
trainer = trainers.UnigramTrainer(vocab_size=20000, max_piece_length=30, shrinking_factor=0.75)
tokenizer.normalizer = normalizers.BertNormalizer()

# train wordpiece tokenizer
# trainer = trainers.WordPieceTrainer(vocab_size=7000)
# tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
# tokenizer.decoder = decoders.WordPiece()
# tokenizer.normalizer = normalizers.BertNormalizer()


# load dataset here
dataset = load_dataset("squad")

# train the tokenizer
tokenizer.train_from_iterator(dataset["train"]["context"], trainer=trainer)

tokenizer.save("unigram_tokenizer.json")

# Load the trained tokenizer from a file
# loaded_tokenizer = Tokenizer.from_file("unigram_tokenizer.json")

loaded_tokenizer = tokenizer
sentence = "Hellene"
output = loaded_tokenizer.encode(sentence)
print(output.tokens)

base_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # studio-ousia/luke-base
# introduce model for question answering (bert)
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

# open unigram_tokenizer.json
vocab = loaded_tokenizer.get_vocab()
min_value = 0
token_list = {}
with open("unigram_tokenizer.json", "r", encoding="utf8") as f:
    data = json.load(f)
    vocabulary = data["model"]["vocab"]
    # get the 10000 most frequent tokens
    for i in range(15000):
        if len(vocabulary[i][0]) > 3:
            token_list[vocabulary[i][0]] = vocabulary[i][1]
            min_value = min(min_value, vocabulary[i][1])

# missing part of the algo which utilizes the vocab log probability (but I am not sure where this is used at all)

vocab = token_list

# save base_tokenizer's vocab to a file
base_tokenizer.save_pretrained("base_tokenizer_data")
# print old vocab size
print(f"Old vocab size: {base_tokenizer.vocab_size}")

# change Ġ occurences in the keys in the vocab to ##
new_vocab = {}
for key in vocab:
    new_key = key.replace("Ġ", "")
    new_vocab[new_key] = vocab[key]


# add the new tokens to the tokenizer
tokens_added = base_tokenizer.add_tokens(list(new_vocab.keys()))
print(f"Added tokens: {tokens_added}")
# print new vocab size
new_vocab_size = base_tokenizer.vocab_size + tokens_added
print(f"New vocab size: {new_vocab_size}")

# increase embedding layer size of model
model.config.vocab_size = new_vocab_size
model.resize_token_embeddings(new_vocab_size)
# get the new embeddings layer
embeddings = model.get_input_embeddings()

# load old tokenizer again
old_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # studio-ousia/luke-base

# adjust the embedding of the new tokens
for token in new_vocab:
    # check if the token is in the vocab
    if token in old_tokenizer.get_vocab():
        continue

    # if not, lookup the token id
    token_id = base_tokenizer.convert_tokens_to_ids(token)

    with torch.no_grad():
        # if not, get the subword embeddings
        subwords = old_tokenizer.tokenize(token)
        subword_ids = old_tokenizer.convert_tokens_to_ids(subwords)
        subword_embeddings = embeddings.weight[subword_ids]
        # set the new token embedding to the mean of the subword embeddings
        # change the token embedding weights to the mean of the subword embeddings weights
        embeddings.weight[token_id] = subword_embeddings.mean(dim=0)
        # print(f"Token: {token}, Token ID: {token_id}, Subwords: {subwords}, Subword IDs: {subword_ids},
        # Subword Embeddings: {subword_embeddings}, New Token Embedding: {embeddings.weight[token_id]}")


# set the new embeddings layer to the model
with torch.no_grad():
    model.set_input_embeddings(embeddings)


# tokenize the dataset
def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = base_tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = base_tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


tokenized_train_dataset = dataset["train"].map(preprocess_training_examples, batched=True,
                                               remove_columns=dataset["train"].column_names)
tokenized_validation_dataset = dataset["validation"].map(preprocess_validation_examples, batched=True,
                                                         remove_columns=dataset["validation"].column_names)

from tqdm.auto import tqdm
import numpy as np

n_best = 20
metric = load_metric("squad")


def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                            end_index < start_index
                            or end_index - start_index + 1 > 512
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0]: offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)


args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    fp16=True,
    per_device_train_batch_size=6,)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    tokenizer=base_tokenizer
)

trainer.train()

