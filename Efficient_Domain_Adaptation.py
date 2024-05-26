##########################################


#This implementation is based on the paper titled "Efficient Domain Adaptation of Language Models via Adaptive Tokenization" by Vin Sachidananda, Jason S. Kessler, and Yi-An Lai, published at SUSTAINLP on September 15, 2021. The primary motivation behind this implementation was to explore how adaptive tokenization can enhance the transfer of pretrained language models to new, domain-specific contexts. By adapting tokenizers, the model can better handle unique vocabularies and linguistic structures of target domains, which is crucial for tasks such as domain-specific information extraction and sentiment analysis.

###############################
# Bibliographic Information
###############################

# DOI: 10.18653/v1/2021.sustainlp-1.16
# Corpus ID: 237513469
# Title: Efficient Domain Adaptation of Language Models via Adaptive Tokenization
# Authors: Vin Sachidananda, Jason S. Kessler, Yi-An Lai
# Publication Date: 15 September 2021
# Published In: SUSTAINLP





#################################


from collections import Counter
from transformers import BertTokenizer,AutoTokenizer
import numpy as np
import unicodedata
from collections import Counter
from transformers import BertTokenizer, BertModel, AutoTokenizer
import torch
from torch import nn
from torch.optim import SGD
import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from tqdm import tqdm, trange

# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def load_documents(dataset_file):
    documents = []
    words = []
    labels = []
    sentence_boundaries = []
    with open(dataset_file) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("-DOCSTART"):
                if words:
                    documents.append(dict(
                        words=words,
                        labels=labels,
                        sentence_boundaries=sentence_boundaries
                    ))
                    words = []
                    labels = []
                    sentence_boundaries = []
                continue

            if not line:
                if not sentence_boundaries or len(words) != sentence_boundaries[-1]:
                    sentence_boundaries.append(len(words))
            else:
                items = line.split(" ")
                words.append(items[0])
                labels.append(items[-1])

    if words:
        documents.append(dict(
            words=words,
            labels=labels,
            sentence_boundaries=sentence_boundaries
        ))
        
    return documents


def load_examples(documents):
    examples = []
    max_token_length = 510
    max_mention_length = 30

    for document in tqdm(documents):
        words = document["words"]
        subword_lengths = [len(tokenizer.tokenize(w)) for w in words]
        total_subword_length = sum(subword_lengths)
        sentence_boundaries = document["sentence_boundaries"]

        for i in range(len(sentence_boundaries) - 1):
            sentence_start, sentence_end = sentence_boundaries[i:i+2]
            if total_subword_length <= max_token_length:
                # if the total sequence length of the document is shorter than the
                # maximum token length, we simply use all words to build the sequence
                context_start = 0
                context_end = len(words)
            else:
                # if the total sequence length is longer than the maximum length, we add
                # the surrounding words of the target sentence　to the sequence until it
                # reaches the maximum length
                context_start = sentence_start
                context_end = sentence_end
                cur_length = sum(subword_lengths[context_start:context_end])
                while True:
                    if context_start > 0:
                        if cur_length + subword_lengths[context_start - 1] <= max_token_length:
                            cur_length += subword_lengths[context_start - 1]
                            context_start -= 1
                        else:
                            break
                    if context_end < len(words):
                        if cur_length + subword_lengths[context_end] <= max_token_length:
                            cur_length += subword_lengths[context_end]
                            context_end += 1
                        else:
                            break

            text = ""
            for word in words[context_start:sentence_start]:
                if word[0] == "'" or (len(word) == 1 and is_punctuation(word)):
                    text = text.rstrip()
                text += word
                text += " "

            sentence_words = words[sentence_start:sentence_end]
            sentence_subword_lengths = subword_lengths[sentence_start:sentence_end]

            word_start_char_positions = []
            word_end_char_positions = []
            for word in sentence_words:
                if word[0] == "'" or (len(word) == 1 and is_punctuation(word)):
                    text = text.rstrip()
                word_start_char_positions.append(len(text))
                text += word
                word_end_char_positions.append(len(text))
                text += " "

            for word in words[sentence_end:context_end]:
                if word[0] == "'" or (len(word) == 1 and is_punctuation(word)):
                    text = text.rstrip()
                text += word
                text += " "
            text = text.rstrip()

            entity_spans = []
            original_word_spans = []
            for word_start in range(len(sentence_words)):
                for word_end in range(word_start, len(sentence_words)):
                    if sum(sentence_subword_lengths[word_start:word_end + 1]) <= max_mention_length:
                        entity_spans.append(
                            (word_start_char_positions[word_start], word_end_char_positions[word_end])
                        )
                        original_word_spans.append(
                            (word_start, word_end + 1)
                        )

            examples.append(dict(
                text=text,
                words=sentence_words,
                entity_spans=entity_spans,
                original_word_spans=original_word_spans,
            ))

    return examples



def is_punctuation(char):
    cp = ord(char)
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


test_documents = load_documents("eng.testb")
test_examples = load_examples(test_documents)
# Load pre-trained BERT tokenizer




def compute_sequence_distribution(corpus, tokenizer, max_length=20, min_freq=1):
    # Initialize a Counter to store sequence frequencies
    seq_counter = Counter()
    
    # Tokenize the corpus and iterate through tokens to create sequences
    for sentence in corpus:
        tokens = tokenizer.tokenize(sentence)
        for i in range(len(tokens)):
            for j in range(1, max_length + 1):
                if i + j <= len(tokens):
                    seq = tuple(tokens[i:i+j])  # Create a sequence as a tuple for hashability
                    seq_counter[seq] += 1
    
    # Filter out sequences that occur less than min_freq times
    seq_counter = {seq: count for seq, count in seq_counter.items() if count >= min_freq}
    
    return seq_counter

# Test dummy sentences
# all_text = [" ".join(doc["words"]) for doc in test_documents]  # Combine words to form text
# base_corpus = ['The quick brown fox jumps over the lazy dog.']  # Example base corpus text

#The BERT model was pretrained on BookCorpus, a dataset consisting of 11,038. Due to the limitations, this test does not include all but only small subset of the corpus
from datasets import load_dataset

#Test on question answerign SQUAD corpus
squad = load_dataset("squad")
def preprocess_squad(data):
    # Extract context and questions to create a corpus
    corpus = []
    for example in data:
        context = example['context']
        question = example['question']
        corpus.append(context + " " + question)  # Combine context and question
    return corpus
squad_corpus = preprocess_squad(squad['train'])
base_corpus=squad_corpus[:10]
# Compute distributions for both corpora
base_seq_distribution = compute_sequence_distribution(base_corpus, tokenizer)
domain_seq_distribution = compute_sequence_distribution(all_text, tokenizer) # all_text represents the domain-specific corpus


# Find sequences that are common to both distributions
common_sequences = [seq for seq in base_seq_distribution if seq in domain_seq_distribution]

# Or if you're working with individual tokens
common_tokens = [token for token in tokenizer.vocab.keys() if token in base_seq_distribution and token in domain_seq_distribution]
# common_tokens = [token for token in base_seq_distribution if token in domain_distribution]
# print("Common tokens:", common_tokens)

# Normalize distributions
total_base = sum(base_seq_distribution.values())
total_domain = sum(domain_seq_distribution.values())
base_seq_probs = {seq: count / total_base for seq, count in base_seq_distribution.items()}
domain_seq_probs = {seq: count / total_domain for seq, count in domain_seq_distribution.items()}

# Calculate Conditional KL Divergence
def calculate_kl_divergence(base_probs, domain_probs):
    kl_divergence = {}
    for seq in base_probs.keys():
        if seq in domain_probs and base_probs[seq] > 0:
            kl_divergence[seq] = domain_probs[seq] * np.log(domain_probs[seq] / base_probs[seq])
    return kl_divergence


# Select token sequences for augmentation
kl_divergence = calculate_kl_divergence(base_seq_probs, domain_seq_probs)
selected_tokens = [token for token, divergence in sorted(kl_divergence.items(), key=lambda item: item[1], reverse=True)]


# Define minimum frequency thresholds
F_min_base = 0.001  
F_min_domain = 0.001  

# Define the maximum number of augmentations and the length of token sequences
N = 100  # Number of augmentations to select
L = 5  # Max length of token sequences to consider for augmentation


augmentations = []

# Sort sequences based on KL divergence
sorted_sequences = sorted(kl_divergence.items(), key=lambda item: item[1], reverse=True)

# Define minimum frequency thresholds
F_min_base = 0.001  
F_min_domain = 0.001  

# Define the maximum number of augmentations and the length of token sequences
N = 100  # Number of augmentations to select
L = 5  # Max length of token sequences to consider for augmentation

# Select token sequences for augmentation based on the criteria
for seq, divergence in sorted_sequences:
    seq_str = ''.join(seq)  # Convert sequence of tokens to string if needed
    if len(augmentations) >= N:
        break
    if len(seq_str) <= L and domain_seq_probs[seq] >= F_min_domain and base_seq_probs[seq] >= F_min_base:
        augmentations.append(seq_str)

# Initialize dummy embeddings for illustration purposes
# In practice, these would be obtained from the actual model's embedding layer
embedding_size = 768  # Typical embedding size for BERT models
num_common_tokens = 100  # Number of common tokens between source and target
num_source_tokens = 50  # Number of source domain-specific tokens
num_target_tokens = 50  # Number of target domain-specific tokens
# Assuming the same embedding sizes and initializations from before
d = 768  # Embedding size for BERT-base
model = BertModel.from_pretrained('bert-base-uncased')

embedding_layer = model.get_input_embeddings()

# Let's also assume `augmentations` contains the domain-specific tokens to be added
new_tokens = augmentations

new_tokens = [tokenizer.tokenize(seq) for seq in augmentations]
new_token_ids = [tokenizer.convert_tokens_to_ids(tok_seq) for tok_seq in new_tokens]

# Flatten the list of token IDs and remove duplicates
flat_new_token_ids = list(set([item for sublist in new_token_ids for item in sublist]))

# Initialize Xt for new tokens with random embeddings (for now)
Xt = torch.randn((len(flat_new_token_ids), embedding_size))

# Retrieve embeddings for common tokens
common_token_ids = [tokenizer.convert_tokens_to_ids(seq) for seq in common_sequences if len(seq) == 1]  # Only single tokens
common_token_ids = [item for sublist in common_token_ids for item in sublist]  # Flatten the list
common_token_ids = list(set(common_token_ids))  # Remove duplicates

# Ensure that we're getting embeddings for individual tokens, not sequences
Cs = embedding_layer.weight[common_token_ids]

# For source-specific tokens (Xs), retrieve their embeddings from the BERT model
# These are tokens that are in BERT's vocabulary but not selected as domain-specific augmentations
source_specific_tokens = [token for token in tokenizer.vocab.keys() if token not in new_tokens]
source_specific_token_ids = tokenizer.convert_tokens_to_ids(source_specific_tokens)
#### NOT SURE ABOUT THIS AS I HAD TO MAKE DIMENSIONS COMPATIBLE:
source_specific_token_ids = [i for i in source_specific_token_ids if i in common_token_ids]
Xs = embedding_layer.weight[source_specific_token_ids]

# Initialize the matrix M with random values
M = torch.randn((d, d), requires_grad=True)
print(M.shape, Xs.shape,torch.matmul(M, Xs.T).shape)
print(Cs.T.shape)
# Define the loss function (Frobenius norm of the difference)
def loss_function(M, Cs, Xs):
    return torch.norm(torch.matmul(M, Xs.T) - Cs.T, p='fro')

# Initialize the optimizer
optimizer = SGD([M], lr=0.01)

# Number of iterations for SGD
num_iterations = 10

# Perform SGD
for _ in range(num_iterations):
    optimizer.zero_grad()  # Zero the gradients
    loss = loss_function(M, Cs, Xs)  # Compute the loss
    loss.backward(retain_graph=True)  # Compute the gradients
    optimizer.step()  # Update M using the gradients

# Now M should be a learned matrix that maps Xs to Cs
# Ct will be initialized using the learned matrix M_hat
Ct = torch.matmul(M, Xt.T).T

# Convert Ct to numpy array if necessary
Ct = Ct.detach().numpy()


# Load pre-trained BERT model
model = BertModel.from_pretrained('bert-base-uncased')

# Add new tokens to the tokenizer
tokenizer.add_tokens(augmentations)

# Resize the model's token embeddings to fit the new extended tokenizer
model.resize_token_embeddings(len(tokenizer))

# Update the model's embeddings with the new initializations
# Assuming Ct contains the initial embeddings for the new tokens
new_token_embeddings = torch.tensor(Ct)

# Get the embeddings matrix from the model
embedding_layer = model.get_input_embeddings()
num_added_tokens = len(augmentations)

# Update the new token embeddings with Ct
embedding_layer.weight.data[-num_added_tokens:, :] = new_token_embeddings
print(new_token_embeddings)