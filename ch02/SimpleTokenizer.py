import re

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("total number of character:", len(raw_text))
print(raw_text[:99])

preprocessed_sample = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed_sample = [item for item in preprocessed_sample if item.strip()]
print(len(preprocessed_sample))
print(preprocessed_sample[:50])

all_words = sorted(set(preprocessed_sample))
vocab_size = len(all_words)
print(vocab_size)

vocab = {token: integer for integer, token in enumerate(all_words)}
print(vocab)


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
Mrs.Gisburn said with pardonable pride.
"""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))

all_tokens = sorted(list(set(preprocessed_sample)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer, token in enumerate(all_tokens)}

print(len(vocab))

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

tokenizer = SimpleTokenizerV1(vocab)
print(tokenizer.encode(text))