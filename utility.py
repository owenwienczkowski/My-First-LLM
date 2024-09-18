# program for running coding exercises
import re


# load the example text file
with open("the-verdict.txt", "r", encoding="utf-8") as f: raw_text = f.read()

'''
# print the total number of characters
print("Total number of character:", len(raw_text))

# print the first 100 characters
print(raw_text[:99])
'''


'''
# basic tokenizer
import re
text = "Hello, world. Is this-- a test?"

# split text into tokens (words or punctuation marks.)
# result = re.split(r'([,.:;?_!"()\']|--|\s)', text)

# OPTIONAL: strip each item in the list of its leading and trailing whitespace (removes redundant whitespace tokens entirely)
result = [item for item in result if item.strip()]

print(result)
'''


# applying the basic tokenizer concept to the sample text

preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)

preprocessed = [item.strip() for item in preprocessed if item.strip()]

# print(len(preprocessed))

# print(preprocessed[:30])

# assign unique token IDs

# take all tokens, convert to a set to have only unique tokens, convert back to a list to sort alphabetically
all_words = sorted(list(set(preprocessed)))

# assign the number of unique tokens to vocab_size
vocab_size = len(all_words)
# print(vocab_size)

# instantiate the dictionary "vocab" to represent the vocabulary of the LLM. Each token is assigned an ID based off of its index
vocab = {token:integer for integer,token in enumerate(all_words)}

'''
# print the vocab
for i, item in enumerate(vocab.items()):
    print(item)

    # stop printing after the first 50 tokens (0 through 49)
    if i >= 49:
        break
'''

# now applying a newly created first iteration of a tokenizer class to complete the above tasks in a more flexible way

from Tokenizers import SimpleTokenizerV1
tokenizer = SimpleTokenizerV1(vocab)


text = """"It's the last he painted, you know," Mrs. Gisburn said with """
ids = tokenizer.encode(text)
# print(ids)

# print(tokenizer.decode(ids))

# this example text will not work without checking for token existence in the encoder as "Hellooo" is not currently part of the vocabulary
# text = "Hellooo, do you like tea?"

tokenizer.encode(text)

# if the check is applied, feel free to run this code block to observe results!! (SPOILER ALERT!! : The word will be missing)
'''
ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))
'''

# list of all tokens, including two new tokens '<|endoftext|>' and <|unk|> which signifiy a separation between two unrelated texts, and an unknown token respectively
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
# print(len(vocab.items()))

# for i, item in enumerate(list(vocab.items())[-5:]):
#     print(item)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
# print(text)

# create a new tokenizer and train on the same initial vocab. Test on new text
from Tokenizers import SimpleTokenizerV2
tokenizer = SimpleTokenizerV2(vocab)

'''
# view the IDs of the tokens in the text
print(tokenizer.encode(text))

# print the tokens based on ID (including the <|unk|> specifically)
print(tokenizer.decode(tokenizer.encode(text)))
'''

### BEGIN BYTE PAIR ENCODING ###
# pip install tiktoken
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
# text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."

# text for testing unknown words to display use of subwords
text = "Akwirw ier"
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)

strings = tokenizer.decode(integers)
# print(strings)

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]

# demo input-target pairs
'''
context_size = 4 #A
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y: {y}")

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

'''

# A dataset for batched inputs and targets
''
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt) #A

        for i in range(0, len(token_ids) - max_length, stride): #B
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

        def __len__(self): #C
            return len(self.input_ids)  
        
        def __getitem__(self, idx): #D
            return self.input_ids[idx], self.target_ids[idx]
        
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2") #A
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) #B
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader) #A
first_batch = next(data_iter)
print(first_batch)
''