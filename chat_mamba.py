import os
import pickle
import time
import jax
import jax.numpy as jnp
from train import MambaLM, TinyShakespeareLoader 

# Configuration
D_MODEL = 256
N_LAYERS = 4
BLOCK_SIZE = 128
MODEL_PATH = "mamba_shakespeare_final.pkl"
LOG_FILE = "electronio_session_logs.txt"

# Sampling Parameters
TEMPERATURE = 0.8  
TOP_K = 10         

def load_model():
    print(f"--- Phase 6: Loading ELECTRONIO (Logging Enabled) ---")
    loader = TinyShakespeareLoader(block_size=BLOCK_SIZE)
    model = MambaLM(vocab_size=loader.vocab_size, d_model=D_MODEL, n_layers=N_LAYERS)
    try:
        with open(MODEL_PATH, "rb") as f:
            params = pickle.load(f)
        print(f"Successfully loaded weights from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: {MODEL_PATH} not found.")
        return None, None, None
    return model, params, loader

def sample_with_top_k(logits, key, temp=1.0, k=10):
    logits = logits / temp
    top_k_logits, top_k_indices = jax.lax.top_k(logits, k)
    sample_idx = jax.random.categorical(key, top_k_logits)
    return top_k_indices[sample_idx]

def chat():
    model, params, loader = load_model()
    if not model: return

    rng = jax.random.PRNGKey(int(time.time()))
    
    # Open log file in append mode
    with open(LOG_FILE, "a") as log:
        log.write(f"\n{'='*50}\nSession: {time.ctime()}\n{'='*50}\n")
        
        print("\nELECTRONIO IS READY. (Logs saving to: " + LOG_FILE + ")")
        print("="*40)

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit']: break
            
            log.write(f"Prompt: {user_input}\nResponse: ")
            print("\nELECTRONIO:", end=" ", flush=True)
            
            input_ids = jnp.array([loader.encode(user_input)])
            curr_seq = input_ids
            
            for _ in range(150):
                rng, subkey = jax.random.split(rng)
                logits = model.apply({'params': params}, curr_seq, train=False)
                last_token_logits = logits[0, -1, :]
                
                next_token_id = sample_with_top_k(last_token_logits, subkey, temp=TEMPERATURE, k=TOP_K)
                char = loader.decode([int(next_token_id)])
                
                print(char, end="", flush=True)
                log.write(char) # Save to log
                
                next_token_reshaped = jnp.array([[next_token_id]])
                curr_seq = jnp.concatenate([curr_seq, next_token_reshaped], axis=1)
                
                if curr_seq.shape[1] > BLOCK_SIZE:
                    curr_seq = curr_seq[:, -BLOCK_SIZE:]

            log.write("\n\n")
            print("\n" + "-"*40)

if __name__ == "__main__":
    chat()