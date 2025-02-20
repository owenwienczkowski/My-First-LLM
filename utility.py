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
# begin basic self-attention mechanism
inputs = torch.tensor(
[[0.43, 0.15, 0.89], # Your (x^1)
[0.55, 0.87, 0.66], # journey (x^2)
[0.57, 0.85, 0.64], # starts (x^3)
[0.22, 0.58, 0.33], # with (x^4)
[0.77, 0.25, 0.10], # one (x^5)
[0.05, 0.80, 0.55]] # step (x^6)
)
'''
# example of determining attention for second word (inputs[1])
query = inputs[1] #A
# calculate attention scores using dot product
# greater dot product infers greater alignment/similarity between elements (how muuch two elements "attend" to one another0). 
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
# print(attn_scores_2)

# normalize the attention scores to get attention weights
# for a basic way to do this, divide by sum of each attention score by the summ of all attention scores
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
# print("Sum:", attn_weights_2_tmp.sum())

# a better method of normaization is the softmax function
# self-defined
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)
attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# pytorch defined
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
# print("Attention weights:", attn_weights_2)
# print("Sum:", attn_weights_2.sum())

# calculate context vector by multiplying embedded input tokens with corresponding attention weights and then summing the vectors
# example still for second word
query = inputs[1] # 2nd input token is the query
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
# print(context_vec_2)
'''
# begin improved self-attention mechanism

# compute the dot products for all pairs of inputs
'''
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores) # unnormalized attention scores

# normalize the attention scores for each input (row) to acquire attention weights
attn_weights = torch.softmax(attn_scores, dim=1)
print(attn_weights)

# print("All row sums:", attn_weights.sum(dim=1))

# using attention weights, compute context vectors through matrix multiplicaiton
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)
'''

# add trainable weights for LLM to learn from, "scaled dot-product attention"

x_2 = inputs[1] #A
d_in = inputs.shape[1] #B
d_out = 2 #C

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
# print(query_2)

keys = inputs @ W_key
values = inputs @ W_value
# print("keys.shape:", keys.shape)
# print("values.shape:", values.shape)

keys_2 = keys[1] #A
attn_score_22 = query_2 @ keys_2
# print(attn_score_22)

attn_scores_2 = query_2 @ keys.T # All attention scores for given query
# print(attn_scores_2)

# normalize the scores via softmax function
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1) # divide by square root of the embedding dimension of the keys
# print(attn_weights_2)

# compute the context vectors
context_vec_2 = attn_weights_2 @ values
# print(context_vec_2)



# A compact self-attention class

import torch.nn as nn
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    # compute and normalize attention scores
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
# print(sa_v1(inputs))

# second class using nn.Module for nn.Linear layers and optimized weight initialization scheme
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
# print(sa_v2(inputs))

'''
# The task is to correctly assign the weights from an instance of
# SelfAttention_v2 to an instance of SelfAttention_v1. To do this, you need
# to understand the relationship between the weights in both versions. (Hint:
# nn.Linear stores the weight matrix in a transposed form.) After the
# assignment, you should observe that both instances produce the same outputs.

# assign each weight of V1 to the transpose of the weights of V2
sa_v1.W_key.data = sa_v2.W_key.weight.T.clone()
sa_v1.W_query.data = sa_v2.W_query.weight.T.clone()
sa_v1.W_value.data = sa_v2.W_value.weight.T.clone()

# print shape of v2, v2 transpose, and (new) v1
print("sa_v2.W_key.weight shape:", sa_v2.W_key.weight.shape)
print("Transposed shape:", sa_v2.W_key.weight.T.shape)
print("sa_v1.W_key.data shape (after assignment):", sa_v1.W_key.data.shape)

output_v1 = sa_v1(inputs)
output_v2 = sa_v2(inputs)

# The function torch.allclose(tensor1, tensor2, atol=some_value) checks if all elements in tensor1 and tensor2 are numerically close within a specified tolerance.
print("Weight transfer successful:", torch.allclose(sa_v1.W_key, sa_v2.W_key.weight.T, atol=1e-6))
print("Outputs match:", torch.allclose(output_v1, output_v2, atol=1e-6))
'''

# Using casual/masked attention to hide future words, predicting off of only past and current words

# process of initialize scores, normalize them, apply a mask, and renormalize
queries = sa_v2.W_query(inputs) #A
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)
# print(attn_weights)

# create a mask matrix to hide future values
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
# print(mask_simple)

# multiply the current weights by the mask to apply future value hiding
masked_simple = attn_weights*mask_simple
# print(masked_simple)

# renormalize the attention weights
row_sums = masked_simple.sum(dim=1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
# print(masked_simple_norm)

# more efficient mask: initialize scores, apply a mask, then normalize

# create a mask where all values above the diagonal are 1
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
# convert the mask to booolean values (1: True, 0: False). Convert false values to -inf
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
# print(masked)

# normalize the attention scores to get the attention weights
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
# print(attn_weights)

# check to see if values are consistent
# print(masked_simple_norm)
# print("Outputs match:", torch.allclose(attn_weights, masked_simple_norm, atol=1e-6))

# application of dropout after implementation of attetnion weights
dropout = torch.nn.Dropout(0.2)
torch.manual_seed(123)
# print(dropout(attn_weights))

# duplicate text to simulate batch input
batch = torch.stack((inputs, inputs), dim=0) # two input texts with 6 tokens each. Each token is a 3D embedding vector


# Casual/Masked attention classclass CausalAttention(nn.Module):
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) #A
        self.register_buffer('mask',torch.triu(torch.ones(context_length, context_length),diagonal=1)) #B
    def forward(self, x):
        b, num_tokens, d_in = x.shape #C b=number of batches, num_tokens=Sequence length , d_in=Feature dimension per token (input embedding size)
# New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2) #C
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
# print("context_vecs.shape:", context_vecs.shape)

# implementing multi-head attention

# trivial method: stacking multiple single head attention modules
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

# example of use of multi-head attention model
torch.manual_seed(123)
context_length = batch.shape[1] # This is the number of tokens
d_in, d_out, num_heads = 3, 2, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads)
context_vecs = mha(batch)
# print(context_vecs)
# print("context_vecs.shape:", context_vecs.shape)

# Exercise 3.2:
# Change the input arguments for the MultiHeadAttentionWrapper(...,
# num_heads=2) call such that the output context vectors are 2-dimensional
# instead of 4-dimensional while keeping the setting num_heads=2. Hint: You
# don't have to modify the class implementation; you just have to change one of
# the other input arguments.
'''
torch.manual_seed(123)
context_length = batch.shape[1] # This is the number of tokens
d_in, d_out, num_heads = 3, 1, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
'''

# computing the outputs for all attention heads simultaneously via matrix multiplication
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads #A
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) #B
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x) #C
        queries = self.W_query(x) #C
        values = self.W_value(x) #C
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) #D
        values = values.view(b, num_tokens, self.num_heads, self.head_dim) 
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.transpose(1, 2) #E
        queries = queries.transpose(1, 2) #E
        values = values.transpose(1, 2) #E
        attn_scores = queries @ keys.transpose(2, 3) #F
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] #G
        attn_scores.masked_fill_(mask_bool, -torch.inf) #H
        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2) #I
        #J
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) #K
        return context_vec 
    
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573], #A
[0.8993, 0.0390, 0.9268, 0.7388],
[0.7179, 0.7058, 0.9156, 0.4340]],
[[0.0772, 0.3565, 0.1479, 0.5331],
[0.4066, 0.2318, 0.4545, 0.9737],
[0.4606, 0.5159, 0.4220, 0.5786]]]])

# print(a @ a.transpose(2, 3))

torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
# print(context_vecs)
# print("context_vecs.shape:", context_vecs.shape)

'''
# Using the MultiHeadAttention class, initialize a multi-head attention
# module that has the same number of attention heads as the smallest GPT-2
# model (12 attention heads). Also ensure that you use the respective input and
# output embedding sizes similar to GPT-2 (768 dimensions). Note that the
# smallest GPT-2 model supports a context length of 1024 tokens.
d_in_challenge, d_out_challenge, context_length_challenge = 768, 768, 1024
mha_challenge = MultiHeadAttention(d_in_challenge, d_out_challenge, context_length_challenge, 0.0, 12)

batch_challenge = torch.rand(2, context_length_challenge, d_in_challenge)
context_vecs_challenge = mha_challenge(batch_challenge)
print(context_vecs_challenge)
print("context_vecs_challenge.shape:", context_vecs_challenge.shape)
'''

# Implementing a GPT model from Scratch To Generate Text
# Coding an LLM architecture:

# Specify the configuration of a small GPT-2 model
GPT_CONFIG_124M = {
"vocab_size": 50257, # Vocabulary size
"context_length": 1024, # Context length
"emb_dim": 768, # Embedding dimension
"n_heads": 12, # Number of attention heads
"n_layers": 12, # Number of layers
"emb_drop_rate": 0.1, # Embedding dropout rate
"short_drop_rate": 0.15, # Shortcut dropout rate
"multi_drop_rate": 0.175, # Multi-Head attention dropout rate
"qkv_bias": False # Query-Key-Value bias
}

# a placeholder GPT backbone

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["emb_drop_rate"])
        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]) 
        self.final_norm = DummyLayerNorm(cfg["emb_dim"]) #B
        self.out_head = nn.Linear(
        cfg["emb_dim"], cfg["vocab_size"], bias=False)
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
class DummyTransformerBlock(nn.Module): #C
    def __init__(self, cfg):
        super().__init__()
    def forward(self, x): #D
        return x
class DummyLayerNorm(nn.Module): #E
    def __init__(self, normalized_shape, eps=1e-5): #F
        super().__init__()
    def forward(self, x):
        return x

# tokenize a batch consisting of two text inputs for the GPT model using the tiktoken tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
# print(batch)

# initialize a new 124 million parameter DummyGPTModel instance and feed it the tokenized batch
torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
# print("Output shape:", logits.shape)
# print(logits)

# normalize layer outputs with ReLU
torch.manual_seed(123)
batch_example = torch.randn(2, 5) #A
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
# print(out)

# examine normalization
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
# print("Mean:\n", mean)
# print("Variance:\n", var)

# apply layer normalization to the layer outputs
out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
# print("Normalized layer outputs:\n", out_norm)
torch.set_printoptions(sci_mode=False)
# print("Mean:\n", mean)
# print("Variance:\n", var)

# layer normalization class
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 # small constant to prevent dividing by zero 
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
# print("Mean:\n", mean)
# print("Variance:\n", var)

# Implementing a feed forward network with GELU activations
# Implementation of the GELU activation function:
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

# compare GELU vs ReLU
import matplotlib.pyplot as plt
# gelu, relu = GELU(), nn.ReLU()
# x = torch.linspace(-3, 3, 100) #A
# y_gelu, y_relu = gelu(x), relu(x)
# plt.figure(figsize=(8, 3))
# for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
#     plt.subplot(1, 2, i)
#     plt.plot(x, y)
#     plt.title(f"{label} activation function")
#     plt.xlabel("x")
#     plt.ylabel(f"{label}(x)")
#     plt.grid(True)
# plt.tight_layout()
# plt.show()

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), GELU(), nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),)
    def forward(self, x):
        return self.layers(x)
    
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768) #A
out = ffn(x)
# print(out.shape)

# Adding shortcut connections

# Example Deep NN to illustrate
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        # Implement 5 layers
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())])
    def forward(self, x):
        for layer in self.layers:
        # Compute the output of the current layer
            layer_output = layer(x)
            # Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123) # specify random seed for the initial weights
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)

def print_gradients(model, x):
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.]])
    # Calculate loss based on how close the target
    # and output are
    loss = nn.MSELoss()
    loss = loss(output, target)
    # Backward pass to calculate the gradients
    loss.backward()
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Print the mean absolute gradient of the weights
            # print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
            pass

print_gradients(model_without_shortcut, sample_input)

torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(
layer_sizes, use_shortcut=True)
print_gradients(model_with_shortcut, sample_input)

# Connecting attention and linear layers in a transformer block

# Transformer block component: 
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
        d_in=cfg["emb_dim"],
        d_out=cfg["emb_dim"],
        context_length=cfg["context_length"],
        num_heads=cfg["n_heads"],
        dropout=cfg["multi_drop_rate"],
        qkv_bias=cfg["qkv_bias"])
        # layers normalized then dropout applied
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["short_drop_rate"])
    def forward(self, x):
        #A
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut # Add the original input back
        shortcut = x #B
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut #C
        return x
    
torch.manual_seed(123)
x = torch.rand(2, 4, 768) #A
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
# print("Input shape:", x.shape)
# print("Output shape:", output.shape)

# Coding the GPT model
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["emb_drop_rate"])
        self.trf_blocks = nn.Sequential(
        *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
        cfg["emb_dim"], cfg["vocab_size"], bias=False)
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        #A
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
# print("Input batch:\n", batch)
# print("\nOutput shape:", out.shape)
# print(out)

total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params:,}")

# print("Token embedding layer shape:", model.tok_emb.weight.shape)
# print("Output layer shape:", model.out_head.weight.shape)

total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
# print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

# compute the memory requirements of the 163 million parameters 
total_size_bytes = total_params * 4 #A
total_size_mb = total_size_bytes / (1024 * 1024) #B
# print(f"Total size of the model: {total_size_mb:.2f} MB")

'''
# Initializing larger GPT models

# Without making any code modifications besides
# updating the configuration file, use the GPTModel class to implement GPT-2
# medium (using 1024-dimensional embeddings, 24 transformer blocks, 16
# multi-head attention heads), GPT-2 large (1280-dimensional embeddings, 36
# transformer blocks, 20 multi-head attention heads), and GPT-2 XL (1600-
# dimensional embeddings, 48 transformer blocks, 25 multi-head attention
# heads). As a bonus, calculate the total number of parameters in each GPT model.

# Specify the configuration of a medium GPT-2 model
GPT_CONFIG_MED = {
"vocab_size": 50257, # Vocabulary size
"context_length": 1024, # Context length
"emb_dim": 1024, # Embedding dimension
"n_heads": 16, # Number of attention heads
"n_layers": 24, # Number of layers
"emb_drop_rate": 0.1, # Embedding dropout rate
"short_drop_rate": 0.15, # Shortcut dropout rate
"multi_drop_rate": 0.175, # Multi-Head attention dropout rate
"qkv_bias": False # Query-Key-Value bias
}

# Specify the configuration of a large GPT-2 model
GPT_CONFIG_LARGE = {
"vocab_size": 50257, # Vocabulary size
"context_length": 1024, # Context length
"emb_dim": 1280, # Embedding dimension
"n_heads": 20, # Number of attention heads
"n_layers": 36, # Number of layers
"emb_drop_rate": 0.1, # Embedding dropout rate
"short_drop_rate": 0.15, # Shortcut dropout rate
"multi_drop_rate": 0.175, # Multi-Head attention dropout rate
"qkv_bias": False # Query-Key-Value bias
}

# Specify the configuration of an XL GPT-2 model
GPT_CONFIG_XL = {
"vocab_size": 50257, # Vocabulary size
"context_length": 1024, # Context length
"emb_dim": 1600, # Embedding dimension
"n_heads": 25, # Number of attention heads
"n_layers": 48, # Number of layers
"emb_drop_rate": 0.1, # Embedding dropout rate
"short_drop_rate": 0.15, # Shortcut dropout rate
"multi_drop_rate": 0.175, # Multi-Head attention dropout rate
"qkv_bias": False # Query-Key-Value bias
}

med_model = GPTModel(GPT_CONFIG_MED)
large_model = GPTModel(GPT_CONFIG_LARGE)
xl_model = GPTModel(GPT_CONFIG_XL)

# medium model params
total_params_med = sum(p.numel() for p in med_model.parameters())
print(f"Total number of parameters in medium: {total_params_med:,}")

total_params_gpt2_med = total_params_med - sum(p.numel() for p in med_model.out_head.parameters())
print(f"Number of trainable parameters considering weight tying in medium model: {total_params_gpt2_med:,}")

# compute the memory requirements of the 406,212,608 parameters 
total_size_bytes_med = total_params_med * 4 #A
total_size_mb_med = total_size_bytes_med / (1024 * 1024) #B
print(f"Total size of the medium model: {total_size_mb_med:.2f} MB")

# large model params
total_params_large = sum(p.numel() for p in large_model.parameters())
print(f"Total number of parameters in large: {total_params_large:,}")

total_params_gpt2_large = total_params_large - sum(p.numel() for p in large_model.out_head.parameters())
print(f"Number of trainable parameters considering weight tying in large model: {total_params_gpt2_large:,}")

# compute the memory requirements of the 773,891,840 parameters 
total_size_bytes_large = total_params_large * 4 #A
total_size_mb_large = total_size_bytes_large / (1024 * 1024) #B
print(f"Total size of the large model: {total_size_mb_large:.2f} MB")

# xl model params
total_params_xl = sum(p.numel() for p in xl_model.parameters())
print(f"Total number of parameters in xl: {total_params_xl:,}")

total_params_gpt2_xl = total_params_xl - sum(p.numel() for p in xl_model.out_head.parameters())
print(f"Number of trainable parameters considering weight tying in xl model: {total_params_gpt2_xl:,}")

# compute the memory requirements of the 1,557,380,800 parameters 
total_size_bytes_xl = total_params_xl * 4 #A
total_size_mb_xl = total_size_bytes_xl / (1024 * 1024) #B
print(f"Total size of the xl model: {total_size_mb_xl:.2f} MB")
'''

# Generating text

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]  

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

# sample generation
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
# print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0) #A
# print("encoded_tensor.shape:", encoded_tensor.shape)

model.eval() #A
out = generate_text_simple(model=model, idx=encoded_tensor, max_new_tokens=6, context_size=GPT_CONFIG_124M["context_length"])
# print("Output:", out)
# print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
# print(decoded_text)

# Pretraining on Unlabeled Data

GPT_CONFIG_124M = {
"vocab_size": 50257,
"context_length": 256, #A
"emb_dim": 768,
"n_heads": 12,
"n_layers": 12,
"emb_drop_rate": 0.1, #B
"short_drop_rate": 0.1, #B
"multi_drop_rate": 0.1, #B
"drop_rate": 0.1, #B
"qkv_bias": False
}
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

# text to token id conversion
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())
start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(model=model, idx=text_to_token_ids(start_context, tokenizer), max_new_tokens=10, context_size=GPT_CONFIG_124M["context_length"])
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# Calculating the text generation loss
inputs = torch.tensor(
[[16833, 3626, 6100], # ["every effort moves",
[40, 1107, 588]]) # "I really like"]

targets = torch.tensor(
[[3626, 6100, 345 ], # [" effort moves you",
[588, 428, 11311]]) # " really like chocolate"]

with torch.no_grad(): #A
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1) # Probability of each token in vocabprint(probas.shape)
# print(probas.shape) # 2x3x50257 = batch size x tokens per input/batch x vocab size

token_ids = torch.argmax(probas, dim=-1, keepdim=True)
# print("Token IDs:\n", token_ids)

# print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
# print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

# implement the text evaluation function

text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
# print("Text 1:", target_probas_1)
text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
# print("Text 2:", target_probas_2)

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
# print(log_probas)

avg_log_probas = torch.mean(log_probas)
# print(avg_log_probas)

neg_avg_log_probas = avg_log_probas * -1
# print(neg_avg_log_probas)

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
# print("Flattened logits:", logits_flat.shape)
# print("Flattened targets:", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
# print(loss)

perplexity = torch.exp(loss)

# Calculating the training and validation set losses
file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
# print("Characters:", total_characters)
# print("Tokens:", total_tokens)

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,   
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
)

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if num_batches is None:
        num_batches = len(data_loader) #A
    else:
        num_batches = min(num_batches, len(data_loader)) #B
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item() #C
        else:
            break
    return total_loss / num_batches #D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #A
model.to(device)
# print("device:",device)
train_loss = calc_loss_loader(train_loader, model, device) #B
val_loss = calc_loss_loader(val_loader, model, device)
# print("Training loss:", train_loss)
# print("Validation loss:", val_loss)

# Training an LLM
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

# torch.manual_seed(123)
# model = GPTModel(GPT_CONFIG_124M)
# model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0)
# num_epochs = 10
# train_losses, val_losses, tokens_seen = train_model_simple(
#     model, train_loader, val_loader, optimizer, device,
#     num_epochs=num_epochs, eval_freq=5, eval_iter=1,
#     start_context="Every effort moves you", tokenizer=tokenizer
# )

# create a graph of loss over epochs
from matplotlib.ticker import MaxNLocator
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()

# epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# Decoding strategies to control randomness

# transfer the model back from the GPU to the CPU since
# inference with a relatively small model does not require a GPU
# model.to("cpu")
# model.eval()

# tokenizer = tiktoken.get_encoding("gpt2")
# token_ids = generate_text_simple(
#     model=model,
#     idx=text_to_token_ids("Every effort moves you", tokenizer),
#     max_new_tokens=25,
#     context_size=GPT_CONFIG_124M["context_length"]
# )
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# Temperature scaling

vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}
inverse_vocab = {v: k for k, v in vocab.items()}

next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()
# print(inverse_vocab[next_token_id])

torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
# print(inverse_vocab[next_token_id])

def print_sampled_tokens(probas):
    torch.manual_seed(123) # Manual seed for reproducibility
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

# print_sampled_tokens(probas)

def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

temperatures = [1, 0.1, 5] # Original, higher, and lower temperature
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
# x = torch.arange(len(vocab))
# bar_width = 0.15
# fig, ax = plt.subplots(figsize=(5, 3))
# for i, T in enumerate(temperatures):
#     rects = ax.bar(x + i * bar_width, scaled_probas[i],
#         bar_width, label=f'Temperature = {T}')
# ax.set_ylabel('Probability')
# ax.set_xticks(x)
# ax.set_xticklabels(vocab.keys(), rotation=90)
# ax.legend()
# plt.tight_layout()
# plt.show()


# Exercise 5.1
# Use the print_sampled_tokens function to print the sampling frequencies of
# the softmax probabilities scaled with the temperatures shown in Figure 5.13.
# How often is the word "pizza" sampled in each case? Can you think of a
# faster and more accurate way to determine how often the word "pizza" is sampled?

# for _ in range(len(temperatures)):
#     print_sampled_tokens(scaled_probas[_])
#     print(scaled_probas[_][6])
#     # print(inverse_vocab[6])


#  top-k sampling
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
# print("Top logits:", top_logits)
# print("Top positions:", top_pos)

new_logits = torch.where(
    condition=next_token_logits < top_logits[-1], #A
    input=torch.tensor(float('-inf')), #B
    other=next_token_logits #C
)
# print(new_logits)

topk_probas = torch.softmax(new_logits, dim=0)
# print(topk_probas)

# implement combination of temperature and top-k samplilng into generate function
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

torch.manual_seed(123)

# token_ids = generate(
#     model=model,
#     idx=text_to_token_ids("Every effort moves you", tokenizer),
#     max_new_tokens=15,
#     context_size=GPT_CONFIG_124M["context_length"],
#     top_k=25,
#     temperature=1.4
# )

# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# Exercise 5.2
# Play around with different temperatures and top-k settings. Based on your
# observations, can you think of applications where lower temperature and topk settings are desired? Vice versa, can you think of applications where higher
# temperature and top-k settings are preferred? (It's recommended to also
# revisit this exercise at the end of the chapter after loading the pretrained
# weights from OpenAI.)

# token_ids = generate(
#     model=model,
#     idx=text_to_token_ids("Every effort moves you", tokenizer),
#     max_new_tokens=15,
#     context_size=GPT_CONFIG_124M["context_length"],
#     top_k=3,
#     temperature=0.09
# )

# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# token_ids = generate(
#     model=model,
#     idx=text_to_token_ids("Every effort moves you", tokenizer),
#     max_new_tokens=15,
#     context_size=GPT_CONFIG_124M["context_length"],
#     top_k=30,
#     temperature=4.5
# )

# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# Lower values of topk and temperature may be more desirable in situation where highly specific/accurate responses are critical
# Higher values of topk and temperature may be more valuable in creative writing or storytelling circumstances


# Exercise 5.3
# What are the different combinations of settings for the generate function to
# force deterministic behavior, that is, disabling the random sampling such that
# it always produces the same outputs similar to the generate_simple function?

# the settings to disable random sampling are related to both temperature and topk
# we can produce the same output as the generate_simple function through these settings:
# topk = None
# temperature = 1
# the code with these settings is below
# token_ids = generate(
#     model=model,
#     idx=text_to_token_ids("Every effort moves you", tokenizer),
#     max_new_tokens=15,
#     context_size=GPT_CONFIG_124M["context_length"],
#     top_k=None,
#     temperature=1
# )
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# An alternative method is to set topk=1. This will have only the most likely token generated



# Loading and saving model weights
# save model state
torch.save(model.state_dict(), "model.pth")

model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# save model state and optimizer state
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    },
    "model_and_optimizer.pth"
)

checkpoint = torch.load("model_and_optimizer.pth", weights_only=True)

model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# model.train()


#  Loading pretrained weights from OpenAI


# If the download code does not work for you, it could be due to intermittent
# internet connection, server issues, or changes in how OpenAI shares the
# weights of the open-source GPT-2 model. In this case, please visit this
# online code repository at https://github.com/rasbt/LLMs-fromscratch for alternative and updated instructions, and please reach out via the
# Manning Forum for further questions.

# import urllib.request
# url = (
#     "https://raw.githubusercontent.com/rasbt/"
#     "LLMs-from-scratch/main/ch05/"
#     "01_main-chapter-code/gpt_download.py"
# )
# filename = url.split('/')[-1]
# urllib.request.urlretrieve(url, filename)

from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

# print("Settings:", settings)
# print("Parameter dictionary keys:", params.keys())
# print(params["wte"])
# print("Token embedding weight tensor dimensions:", params["wte"].shape)

model_configs = {
"gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
"gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},"gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
"gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])

NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})

gpt = GPTModel(NEW_CONFIG)
gpt.eval()

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

# Loading OpenAI weights into our GPT model code

import numpy as np

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe']) #A
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    for b in range(len(params["blocks"])): #B
        q_w, k_w, v_w = np.split( #C
        (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
        gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
        gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
        gpt.trf_blocks[b].att.W_value.weight, v_w.T)
        q_b, k_b, v_b = np.split(
        (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
        gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
        gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
        gpt.trf_blocks[b].att.W_value.bias, v_b)
        gpt.trf_blocks[b].att.out_proj.weight = assign(
        gpt.trf_blocks[b].att.out_proj.weight,
        params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
        gpt.trf_blocks[b].att.out_proj.bias,
        params["blocks"][b]["attn"]["c_proj"]["b"])
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
        gpt.trf_blocks[b].ff.layers[0].weight,
        params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
        gpt.trf_blocks[b].ff.layers[0].bias,
        params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
        gpt.trf_blocks[b].ff.layers[2].weight,
        params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
        gpt.trf_blocks[b].ff.layers[2].bias,
        params["blocks"][b]["mlp"]["c_proj"]["b"])
        gpt.trf_blocks[b].norm1.scale = assign(
        gpt.trf_blocks[b].norm1.scale,
        params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
        gpt.trf_blocks[b].norm1.shift,
        params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
        gpt.trf_blocks[b].norm2.scale,
        params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
        gpt.trf_blocks[b].norm2.shift,
        params["blocks"][b]["ln_2"]["b"])
        
gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"]) #D

load_weights_into_gpt(gpt, params)
gpt.to(device)

torch.manual_seed(123)
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# Exercise 5.5
# Calculate the training and validation set losses of the GPTModel with the
# pretrained weights from OpenAI on the "The Verdict" dataset.

train_loss = calc_loss_loader(train_loader, gpt, device)
val_loss = calc_loss_loader(val_loader, gpt, device)
