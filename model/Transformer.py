import jax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from ._base import BaseModel


# Define Multi-head self-attention mechanism in JAX
class MultiHeadSelfAttention(BaseModel):
    '''
    ## Multi-head self-attention mechanism
    ### Args:
        - num_heads: int, number of attention heads
        - embed_size: int, size of the input embeddings, must be divisible by num_heads
        - masked: bool, whether to use masked self-attention
    '''
    num_heads: int
    embed_size: int
    masked: bool

    def setup(self):
        self.qkv = nn.Dense(self.embed_size * 3)
        self.output = nn.Dense(self.embed_size)
    def _generate_square_subsequent_mask(self, seq_length, num_heads):
        if self.masked:
            mask=np.triu(np.ones((num_heads, seq_length, seq_length)),1)
            mask[mask==1] = float('-inf')
            mask=jnp.array(mask)
            return mask
        else:
            return jnp.zeros((num_heads, seq_length, seq_length))
    def __call__(self, x):
        batch_size, seq_length, embed_size = x.shape
        _mask=self._generate_square_subsequent_mask(seq_length, self.num_heads)

        # Compute query, key, value matrices
        qkv = self.qkv(x).reshape(batch_size, seq_length, 3, self.num_heads, self.embed_size // self.num_heads).transpose((2, 0, 3, 1, 4)) # (3, batch_size, num_heads, seq_length, head_dim)
        q, k, v = jnp.split(qkv, 3, axis=0)  # Split along Q, K, V
        q = q.squeeze(0) # (batch_size, num_heads, seq_length, head_dim)
        k = k.squeeze(0)
        v = v.squeeze(0)

        attention_scores = jnp.einsum('bhqd, bhkd -> bhqk', q, k) / jnp.sqrt(self.embed_size)
        attention_scores+=_mask
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        # Attention applied to value
        attention_output = jnp.einsum('bhqk, bhvd -> bhqd', attention_weights, v)
        attention_output = attention_output.reshape(batch_size, seq_length, self.embed_size)

        return self.output(attention_output)

# Define the Transformer block
class TransformerBlock(BaseModel):
    '''
    ## Transformer block
    ### Args:
        - num_heads: int, number of attention heads
        - embed_size: int, size of the input embeddings, must be divisible by num_heads
        - ffn_dim: int, size of the hidden layer in the feed-forward network
        - masked: bool, whether to use masked self-attention
        - dropout_rate: float, dropout rate
        - training: bool, whether the model is in training mode, default is True
            If False, dropout is disabled and deterministic is set to True.
            But due to the immutable nature of Flax modules,
            we cannot switch the model.train() and model.eval() as in PyTorch.
            As a result, we need to instantiate a new model with training=False to enable inference mode.
    '''
    num_heads: int
    embed_size: int
    ffn_dim: int
    masked: bool
    dropout_rate: float = 0.1
    training: bool = True

    def setup(self):
        self.attention = MultiHeadSelfAttention(num_heads=self.num_heads, embed_size=self.embed_size, masked=self.masked)
        self.linear_1 = nn.Dense(self.ffn_dim)
        self.linear_2 = nn.Dense(self.embed_size)
        # self.dropout=nn.Dropout(self.dropout_rate,deterministic=not self.training,rng_collection='dropout')
        self.layer_norm_1 = nn.LayerNorm()
        self.layer_norm_2 = nn.LayerNorm()

    def __call__(self, x):
        # Apply self-attention and add residual connection
        attention_output = self.attention(x)
        out1 = self.layer_norm_1(x + attention_output)
        ffn_output = nn.relu(self.linear_1(out1))
        ffn_output = self.linear_2(ffn_output)
        # Apply feed-forward network and add residual connection
        # ffn_output = self.ffn(out1)
        return self.layer_norm_2(out1 + ffn_output)

# Define the full Transformer model
class TransformerModel(BaseModel):
    masked: bool
    num_layers: int
    num_heads: int
    embed_size: int
    ffn_dim: int
    vocab_size: int
    max_length: int
    '''
    ## Transformer model
    ### Args:
        - masked: bool, whether to use masked self-attention
        - num_layers: int, number of transformer blocks
        - num_heads: int, number of attention heads
        - embed_size: int, size of the input embeddings, must be divisible by num_heads
        - ffn_dim: int, size of the hidden layer in the feed-forward network
        - vocab_size: int, size of the vocabulary
        - max_length: int, maximum length of the input sequence
    '''

    def setup(self):
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_size)
        self.pos_embedding = self.param('pos_embedding', jax.nn.initializers.normal(), (self.max_length, self.embed_size))
        self.transformer_blocks = [
            TransformerBlock(num_heads=self.num_heads, embed_size=self.embed_size, ffn_dim=self.ffn_dim, masked=self.masked)
            for _ in range(self.num_layers)
        ]
        self.output=nn.Dense(1)

    def __call__(self, x):
        batch_size, seq_length = x.shape
        # Apply word embedding and position embedding
        # For Ising model, the input is an array of integers, so we can cast it to int32
        x = self.embedding(x.astype(jnp.int32)) + self.pos_embedding[:seq_length]
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        x=self.output(x)
        x=jnp.sum(x.squeeze(-1),axis=-1)
        return x