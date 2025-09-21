# Adaptive Tokenizing

Adaptive Tokenizing is a public implementation of the study Efficient Domain Adaptation of Language Models via Adaptive
Tokenization by Sachidananda et.al. designed to provide efficient and flexible tokenization for various natural language processing (NLP) tasks. It allows for customizable tokenizing strategies, making it suitable for different languages and text formats.

## Features

- **Adaptive Tokenization:** Supports dynamic strategies for splitting text into tokens.
- **Customizable Rules:** Easily configure tokenization rules for specific use cases.
- **Performance-Oriented:** Optimized for speed and low memory footprint.
- **Extensible:** Add your own tokenizing logic with minimal effort.

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'adaptive_tokenizing'
```

And then execute:

```bash
bundle install
```

Or install it yourself as:

```bash
gem install adaptive_tokenizing
```

## Usage

Here’s a basic example:

```ruby
require 'adaptive_tokenizing'

tokenizer = AdaptiveTokenizing::Tokenizer.new
tokens = tokenizer.tokenize("This is an example sentence.")
puts tokens.inspect
```

### Customizing Tokenization

```ruby
tokenizer = AdaptiveTokenizing::Tokenizer.new(strategy: :custom)
tokens = tokenizer.tokenize("Custom tokenization rules here!")
```

For advanced usage and customization, please refer to the documentation or inline comments.

## Paper Summary

This repository is the first public implementation of the paper:

**Efficient Domain Adaptation of Language Models via Adaptive Tokenization**  
Sachidananda et al.

### Motivation

Contextual embedding-based language models like BERT and RoBERTa achieve state-of-the-art results across a variety of NLP tasks. However, their performance drops when fine-tuned on data from domains different from their original pretraining data. Traditional methods to adapt these models to new domains involve costly and time-consuming additional pretraining on domain-specific corpora.

### Proposed Approach

Instead of further pretraining, this paper proposes **adaptive tokenization** as an efficient alternative for domain adaptation. The key ideas are:

- **Domain-Specific Token Selection**: Identify subword sequences that are frequent in the target domain but rare in the base corpus, using divergences in conditional token distributions.
- **Tokenizer Augmentation**: Augment the pretrained tokenizer’s vocabulary with these domain-specific subword tokens.
- **No Further Pretraining Required**: The model is adapted simply by updating the tokenizer and embedding matrix, avoiding the need for expensive retraining.

### Results

- On diverse domain datasets, adaptive tokenization for a pretrained RoBERTa model achieves over 97% of the performance gains obtained by full domain-specific pretraining.
- The approach is substantially more efficient, requiring less training and inference time, and produces smaller models than other tokenizer augmentation methods.
- In experiments, adaptive tokenization incurred only a modest increase in model parameters (about 6% for 10k new tokens), but adaptation was 72x faster than domain-specific pretraining on large compute hardware.

### Novelty

- Selection of domain-associated subword tokens is statistical and efficient.
- Embedding initialization for new tokens does not require further pretraining.
- The approach rivals the accuracy of domain-adaptive pretraining with significantly lower computational cost.

### Related Work

Previous methods (DAPT, TAPT) require further pretraining or add whole words as tokens; these often result in larger models and increased resource requirements. Adaptive tokenization, by contrast, is both effective and resource-efficient.

### Implementation

This repository provides a public Ruby implementation of the adaptive tokenization method, allowing researchers and practitioners to efficiently adapt language models to new domains by simply updating their tokenizers.

For further details, see the full paper:  
[arXiv:2109.07460](https://arxiv.org/abs/2109.07460)
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Authors

- [RubyBit](https://github.com/RubyBit)

## Acknowledgements

- Ruby community
- NLP researchers and developers
