import sentencepiece as spm
import json 
def train():
    spm.SentencePieceTrainer.train(model_prefix="./tokenizer/target_unigram_model/target_18000", input='./tokenizer/target_unigram_model/spm_train_psyqa.txt', model_type="unigram", vocab_size=18000, split_by_whitespace=False, max_sentencepiece_length=32, remove_extra_whitespaces=True,add_dummy_prefix=False)
    # spm.SentencePieceTrainer.train(model_prefix="./tokenizer/target_unigram_model/target_16000", input='./tokenizer/target_unigram_model/spm_train_psyqa.txt', model_type="unigram", vocab_size=16000, split_by_whitespace=False, max_sentencepiece_length=32, remove_extra_whitespaces=True,add_dummy_prefix=False)
    # spm.SentencePieceTrainer.train(model_prefix="./tokenizer/target_unigram_model/target_14000", input='./tokenizer/target_unigram_model/spm_train_psyqa.txt', model_type="unigram", vocab_size=14000, split_by_whitespace=False, max_sentencepiece_length=32, remove_extra_whitespaces=True,add_dummy_prefix=False)
    # spm.SentencePieceTrainer.train(model_prefix="./tokenizer/target_unigram_model/target_10000", input='./tokenizer/target_unigram_model/spm_train_psyqa.txt', model_type="unigram", vocab_size=10000, split_by_whitespace=False, max_sentencepiece_length=32, remove_extra_whitespaces=True,add_dummy_prefix=False)
    # spm.SentencePieceTrainer.train(model_prefix="./tokenizer/target_unigram_model/target_12000", input='./tokenizer/target_unigram_model/spm_train_psyqa.txt', model_type="unigram", vocab_size=12000, split_by_whitespace=False, max_sentencepiece_length=32, remove_extra_whitespaces=True,add_dummy_prefix=False)
    # spm.SentencePieceTrainer.train(model_prefix="./tokenizer/target_unigram_model/target_6000", input='./tokenizer/target_unigram_model/spm_train_psyqa.txt', model_type="unigram", vocab_size=6000, split_by_whitespace=False, max_sentencepiece_length=32, remove_extra_whitespaces=True,add_dummy_prefix=False)
    # spm.SentencePieceTrainer.train(model_prefix="./tokenizer/target_unigram_model/target_4000", input='./tokenizer/target_unigram_model/spm_train_psyqa.txt', model_type="unigram", vocab_size=4000, split_by_whitespace=False, max_sentencepiece_length=32, remove_extra_whitespaces=True,add_dummy_prefix=False)
    # spm.SentencePieceTrainer.train(model_prefix="./tokenizer/target_unigram_model/target_8000", input='./tokenizer/target_unigram_model/spm_train_psyqa.txt', model_type="unigram", vocab_size=8000, split_by_whitespace=False, max_sentencepiece_length=32, remove_extra_whitespaces=True,add_dummy_prefix=False)

    
def test():
    sp = spm.SentencePieceProcessor(model_file='./tokenizer/target_unigram_model/target_8000')
    # print(sp.sample_encode_and_score('在你的问题描述中，你能够感知到自己是一个有目标的人，但是你却缺乏【能量】，这能量更学术一点的来说，就是动机', num_samples=5, alpha=0.1, wor=True, out_type=str))
    # print(sp.encode('在你的问题描述中，你能够感知到自己是一个有目标的人，但是你却缺乏【能量】，这能量更学术一点的来说，就是动机', out_type=str))
    
train()
    
