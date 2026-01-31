import os
# Memory fix must be first
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import time
import json
import pickle  # Added for saving weights
import requests
import numpy as np
import optax
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state

# Make sure mamba_jax.py is in the same folder!
from mamba_jax import MambaBlock 

# =============================================================================
# Phase 3: Tiny Shakespeare Data & Training Pipeline
# =============================================================================

class TinyShakespeareLoader:
    """Handles downloading and tokenizing the Tiny Shakespeare dataset."""
    def __init__(self, block_size=128, batch_size=32, train_split=0.9):
        self.block_size = block_size
        self.batch_size = batch_size
        
        # 1. Download Data
        file_path = 'tinyshakespeare.txt'
        if not os.path.exists(file_path):
            print("Downloading Tiny Shakespeare...")
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            with open(file_path, 'w') as f:
                f.write(requests.get(url).text)
        
        with open(file_path, 'r') as f:
            text = f.read()
            
        # 2. Create Character Vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}
        
        # 3. Encode Data
        data = np.array([self.stoi[c] for c in text], dtype=np.uint16)
        
        # 4. Split Train/Val
        n = int(train_split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        
        print(f"Data loaded. Vocab size: {self.vocab_size}")
        print(f"Train tokens: {len(self.train_data):,}, Val tokens: {len(self.val_data):,}")

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    def get_batch(self, split='train'):
        data = self.train_data if split == 'train' else self.val_data
        # Random offsets
        ix = np.random.randint(0, len(data) - self.block_size, (self.batch_size,))
        x = np.stack([data[i:i+self.block_size] for i in ix])
        y = np.stack([data[i+1:i+self.block_size+1] for i in ix])
        return jnp.array(x), jnp.array(y)

class MambaLM(nn.Module):
    """
    Language Model Head for Mamba.
    Embeds tokens -> Mamba Stack -> LayerNorm -> Linear -> Logits
    """
    vocab_size: int
    d_model: int
    n_layers: int
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2

    @nn.compact
    def __call__(self, x, use_parallel=True, train=True):
        # x: (B, L) (indices)
        
        # 1. Token Embeddings
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)(x)
        
        # 2. Stack Mamba Blocks
        for i in range(self.n_layers):
            x = MambaBlock(
                d_model=self.d_model,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                name=f'mamba_block_{i}'
            )(x, use_parallel=use_parallel)
            # Note: If you add nn.Dropout later, pass 'train' arg here
            
        # 3. Final Norm & Projection
        x = nn.RMSNorm()(x)
        logits = nn.Dense(self.vocab_size)(x)
        
        return logits

def create_train_state(rng, model, learning_rate):
    """Initialize the model parameters and optimizer state."""
    # Dummy input for initialization
    dummy_input = jnp.ones((1, 64), dtype=jnp.int32)
    params = model.init(rng, dummy_input)['params']
    
    # Optimizer (AdamW is standard for Transformers/Mamba)
    tx = optax.adamw(learning_rate)
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

@jax.jit
def train_step(state, x, y):
    """Single training step with Cross Entropy Loss."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x, train=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step(state, x, y):
    """Calculates loss without updating weights (Validation Mode)."""
    logits = state.apply_fn({'params': state.params}, x, train=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return loss

def estimate_loss(state, loader, eval_iters=20):
    """Averages loss over multiple batches to get a stable curve."""
    out = {}
    # Iterate over both splits
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            x, y = loader.get_batch(split)
            loss = eval_step(state, x, y)
            losses.append(loss)
        out[split] = float(np.mean(losses))
    return out

def generate(model, params, tokenizer, prompt, max_new_tokens=100, rng_key=None):
    """Simple greedy generation loop."""
    if rng_key is None:
        rng_key = jax.random.PRNGKey(int(time.time()))
        
    input_ids = jnp.array([tokenizer.encode(prompt)]) 
    
    print(f"\nGenerating from prompt: '{prompt}'")
    print("-" * 40)
    print(prompt, end='', flush=True)
    
    curr_seq = input_ids
    for _ in range(max_new_tokens):
        logits = model.apply({'params': params}, curr_seq, train=False)
        last_logits = logits[:, -1, :]
        next_token = jnp.argmax(last_logits, axis=-1).reshape(1, 1)
        
        token_id = int(next_token[0,0])
        print(tokenizer.decode([token_id]), end='', flush=True)
        
        curr_seq = jnp.concatenate([curr_seq, next_token], axis=1)
        
        # Keep context window manageable 
        if curr_seq.shape[1] > 256: 
             curr_seq = curr_seq[:, -256:]
             
    print("\n" + "-" * 40)

# =============================================================================
# Main Execution
# =============================================================================

def run_phase3_training():
    print("\n" + "="*60)
    print("PHASE 3: TRAINING ON TINY SHAKESPEARE (Safe Mode)")
    print("="*60)
    
    # 1. Hyperparameters (Optimized for M2 Pro 16GB)
    BATCH_SIZE = 32      
    BLOCK_SIZE = 128     
    D_MODEL = 256        
    N_LAYERS = 4         
    LEARNING_RATE = 4e-4 
    MAX_ITERS = 1000     
    EVAL_INTERVAL = 250
    
    # 2. Setup Data
    loader = TinyShakespeareLoader(block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)
    
    # 3. Initialize Model
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    
    model = MambaLM(
        vocab_size=loader.vocab_size,
        d_model=D_MODEL,
        n_layers=N_LAYERS
    )
    
    state = create_train_state(init_key, model, LEARNING_RATE)
    
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"Model initialized with {param_count:,} parameters.")
    print("Starting training loop...\n")
    
    # Storage for plotting
    history = {
        'step': [],
        'train_loss': [],
        'val_loss': []
    }
    
    # 4. Training Loop
    start_time = time.time()
    
    for step in range(MAX_ITERS + 1): # +1 to ensure we catch the final step
        # Fetch batch
        x_batch, y_batch = loader.get_batch('train')
        
        # Update
        state, loss = train_step(state, x_batch, y_batch)
        
        # Logging
        if step % 10 == 0:
            print(f"Step {step:4d} | Batch Loss: {loss:.4f}", end='\r')
            
        # Evaluation & Generation
        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS:
            # Calculate average loss on Train and Validation sets
            losses = estimate_loss(state, loader)
            
            # Store in history
            history['step'].append(step)
            history['train_loss'].append(losses['train'])
            history['val_loss'].append(losses['val'])
            
            dt = time.time() - start_time
            print(f"\n\n--- Step {step} Eval ---")
            print(f"Time: {dt:.2f}s | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
            
            # Generate sample
            generate(model, state.params, loader, prompt="ELECTRON", max_new_tokens=100)
            print("-----------------------")
            
    print("\nTraining Complete!")
    
    # 5. Save History for Plotting
    with open('training_history.json', 'w') as f:
        json.dump(history, f)
    print("History saved to 'training_history.json'. Run plot_graph.py to view.")

    # 6. Save Model Parameters (Pickle) - NEW ADDITION
    model_filename = "mamba_shakespeare_final.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(state.params, f)
    print(f"Model weights saved to '{model_filename}'")
    
    # Final generation
    generate(model, state.params, loader, prompt="The sun ", max_new_tokens=200)

if __name__ == "__main__":
    run_phase3_training()