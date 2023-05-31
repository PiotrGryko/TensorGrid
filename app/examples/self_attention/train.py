import torch  # we use PyTorch: https://pytorch.org
import torch.nn as nn
from torch.nn import functional as F

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))
# let's look at the first 1000 characters
print(text[:1000])

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))  # all unique characters
vocab_size = len(chars)  # size of all unique characters (65 for this dataset)
print(''.join(chars))
print(vocab_size)
n_embd = 32

# tokenization - converting input into integers
# create a mapping from characters to integers
# tiktoken is used with  openai models  https://github.com/openai/tiktoken
'''
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model("gpt-4")
'''

# we need to convert input data into numbers
# here are the simples encode/decode functions
strings_to_integers = {ch: i for i, ch in enumerate(chars)}
integers_to_strings = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [strings_to_integers[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([integers_to_strings[i] for i in l])  # decoder: take a list of integers, output a string

print(encode("hii there"))
print(decode(encode("hii there")))

# let's now encode the entire text dataset and store it into a torch.Tensor


data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)  # torch.Size([1115389]) torch.int64
print(data[:1000])  # the 1000 characters we looked at earier will to the GPT look like this

# Let's now split up the data into train and validation sets
n = int(0.9 * len(data))  # first 90% will be train data, last 10% validation data
train_data = data[:n]
val_data = data[n:]

block_size = 8
print(train_data[:block_size + 1])
# tensor([18, 47, 56, 57, 58,  1, 15, 47, 58])

# transform predicts the next character that follows the given substring
# transform is trained on random chunks of size {block_size} of the input data
x = train_data[:block_size]
y = train_data[1:block_size + 1]
for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
    print(f"when input is {context} the target: {target}")

# generate 4 random chunks of size 8 of input text
# torch.manual_seed value allows to reproduce the random results

torch.manual_seed(1337)
batch_size = 4  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    high = len(data) - block_size
    size = (batch_size,)
    ix = torch.randint(high=high, size=size)
    print("get batch high:{0} size:{1} ix:{2} data:{3}".format(high, size, ix, data))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size):  # batch dimension
    for t in range(block_size):  # time dimension
        context = xb[b, :t + 1]
        target = yb[b, t]
        print(f"when input is {context.tolist()} the target: {target}")

# feed random chunks of input data into neural network
# simplest possible network is BigramLanguageModel


torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd)
        self.position_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd)
        self.language_model_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        print("forward idx:{0}".format(idx.shape))
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_embd = self.token_embedding_table(idx)  # (B,T,C)
        pos_embd = self.position_embedding_table(torch.arange(T))  # ( T,C)
        x = token_embd + pos_embd
        logits = self.language_model_head(x)  # (B,T,C) - (batch, time, channel) (batch=8, time=4, channel=64)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            print("forward, logits: B:{0} T:{1} C:{2}".format(B, T, C))
            logits = logits.view(B * T, C)
            print("forward, formatted logits: B*T:{0} C:{1}".format(B * T, C))
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    # Generate from the model
    # It will generate the next character for the given context
    def generate(self, idx, max_new_tokens):
        print("generate idx:{0} new tokens:{1}".format(idx, max_new_tokens))
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)  # - this calls "forward" function
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities

            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)


def predict_text():
    idx = torch.zeros(size=(1, 1), dtype=torch.long)  # Start with first character 0 is a new line char
    max_new_tokens = 1000
    generated_data = m.generate(idx=idx, max_new_tokens=max_new_tokens)
    print(generated_data)
    print(generated_data[0])
    print(generated_data[0].tolist())
    print(decode(generated_data[0].tolist()))
    print(loss.item())


# predict_text()
#
# # create a PyTorch optimizer
# optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
# batch_size = 32
# m.eval()
# for steps in range(10000):
#     # sample a batch of data
#     xb, yb = get_batch('train')
#
#     # evaluate the loss
#     logits, loss = m(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()
#     print("step {0} loss {1}".format(steps, loss.item()))
# m.train()
# print(loss.item())
# predict_text()

# consider the following toy example:


torch.manual_seed(1337)
B, T, C = 4, 8, 2  # batch, time, channels
x = torch.randn(B, T, C)
print(x.shape)

# We want x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t + 1]  # (t,C)
        xbow[b, t] = torch.mean(xprev, 0)

print(x[0])
print(xbow[0])

# version 2: using matrix multiply for a weighted aggregation
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x  # (B, T, T) @ (B, T, C) ----> (B, T, C)
print(torch.allclose(xbow, xbow2))  # - true means equals

# version 3: use Softmax
tril = torch.tril(torch.ones(T, T))  # triangle matrix, upper right filled with zeros, bottom left filled with ones
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))  # replace zeros with -inf
wei = F.softmax(wei, dim=-1)  # apply soft max, it replaces ones with weights
xbow3 = wei @ x
torch.allclose(xbow, xbow3)

# version 4: use self-attention
torch.manual_seed(1337)
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

# let's see a single head perform self -attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)  # (B,T,16)
q = query(x)  # (B,T,16)

wei = q @ k.transpose(-2, -1)  # (B,T,16) @ (B,16,T) ---> (B,T,T)

tril = torch.tril(torch.ones(T, T))  # triangle matrix, upper right filled with zeros, bottom left filled with ones
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))  # replace zeros with -inf
wei = F.softmax(wei, dim=-1)  # apply soft max, it replaces ones with weights
v = value(x)
out = wei @ v
print(out.shape)

'''Notes:

- Attention is a communication mechanism. Can be seen as nodes in a directed graph looking at each other and aggregating information 
  with a weighted sum from all nodes that point to them, with data-dependent weights.
- There is no notion of space. Attention simply acts over a set of vectors. This is why we need to positionally encode tokens.
- Each example across batch dimension is of course processed completely independently and never "talk" to each other
- In an "encoder" attention block just delete the single line that does masking with tril, allowing all tokens to communicate. 
  This block here is called a "decoder" attention block because it has triangular masking, and is usually used in autoregressive settings, like language 
  modeling.
- "self-attention" just means that the keys and values are produced from the same source as queries. In "cross-attention", the queries still 
  get produced from x, but the keys and values come from some other, external source (e.g. an encoder module)
-"Scaled" attention additional divides wei by 1/sqrt(head_size). This makes it so when input Q,K are unit variance, wei will be unit variance 
  too and Softmax will stay diffuse and not saturate too much. Illustration below'''


