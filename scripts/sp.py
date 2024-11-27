import json
import os
import sentencepiece

home_dir = os.path.expanduser("~")
text_tokenizer = sentencepiece.SentencePieceProcessor(home_dir + "/tmp/tokenizer_spm_32k_3.model")

vocab = {}
for id in range(text_tokenizer.vocab_size()):
    vocab[id] = text_tokenizer.id_to_piece(id)
with open(home_dir + "/tmp/tokenizer_spm_32k_3.json", "w") as json_file:
    json.dump(vocab, json_file)
