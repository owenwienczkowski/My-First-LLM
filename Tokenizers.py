import re

# elementary tokenizer 
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab  # A
        self.int_to_str = {i: s for s, i in vocab.items()}  # B
    
    # text to vocabulary
    def encode(self, text):  # C
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed] # use if s in self.str_to_int to check for if token exists in the vocab.
        return ids
    
    # vocabulary to text
    def decode(self, ids):  # D
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)  # E
        return text
    
# added ability to identify unknown tokens in the encoding stage, replacing them with "<|unk|>"
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int #A
            else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) #B
        return text
