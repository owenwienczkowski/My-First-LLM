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

'''# list of all tokens, including two new tokens '<|endoftext|>' and <|unk|> which signifiy a separation between two unrelated texts, and an unknown token respectively
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


# view the IDs of the tokens in the text
print(tokenizer.encode(text))

# print the tokens based on ID (including the <|unk|> specifically)
print(tokenizer.decode(tokenizer.encode(text)))''
'''

### BEGIN BYTE PAIR ENCODING ###
# pip install tiktoken
import tiktoken
'''
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
# '''
# demo input-target pairs
'''
context_size = 4 #A
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

'''

# A dataset for batched inputs and targets
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
    dataloader = DataLoader( dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


# dataloader to generate batches with input-with pairs
# Data loaders with different strides and context sizes created by adjusting parameter values
'''
dataloader = create_dataloader_v1(raw_text, batch_size=2, max_length=8, stride=2, shuffle=False)
data_iter = iter(dataloader) #A
first_batch = next(data_iter)
print(first_batch)
second_batch = next(data_iter)
print(second_batch)
'''

# use the data loader to sample with a batch size greater than 1
'''
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)
'''
# Creating token embeddings
'''
input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
# print(embedding_layer(torch.tensor([3])))
print(embedding_layer(input_ids))
'''

'''# Encoding word positions
output_dim = 256
vocab_size = 50257
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print("Token embeddings:\n", token_embeddings.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print("Position embeddings:\n",pos_embeddings.shape)


# Input embeddings = token embeddings + position embeddings
input_embeddings = token_embeddings + pos_embeddings
print("Input embeddings:\n",input_embeddings.shape)
'''

# embedded input sentence: 
inputs = torch.tensor(
[[0.43, 0.15, 0.89], # Your (x^1)
[0.55, 0.87, 0.66], # journey (x^2)
[0.57, 0.85, 0.64], # starts (x^3)
[0.22, 0.58, 0.33], # with (x^4)
[0.77, 0.25, 0.10], # one (x^5)
[0.05, 0.80, 0.55]] # step (x^6)
)


# COMPUTE ATTENTION WEIGHTS FOR ONE CONTEXT VECTOR

# calculate the intermediate attention scores between the query token and each input token
# determined by computing the dot product of the query, x(2), with every other input token:
query = inputs[1] #A
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print("Attention scores:", attn_scores_2)

# normalize attention scores to obtain attention weights (that sum to 1)

'''attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum() # more practical to instead use softmax function
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())'''

'''# use naive softmax which is more realistic
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())'''

# use torch softmax for obtain attention weights
'''attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

query = inputs[1] # 2nd input token is the query
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
print(context_vec_2)'''

# COMPUTE ATTENTION WEIGHTS FOR ALL CONTEXT VECTORS SIMULTANEOUSLY

# compute dot products for all pairs of inputs
attn_scores = torch.empty(6, 6)

# multiply using additional for-loop
'''for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
       attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)'''

# multiply the input tensor (matrix) by its transpose to achieve same attention score results more quickly
attn_scores = inputs @ inputs.T
print(attn_scores)

# normalize each row 
attn_weights = torch.softmax(attn_scores, dim=1)
print(attn_weights)

# verify rows sum to 1
'''row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=1))'''

# compute all context vectors via matrix multiplication
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

