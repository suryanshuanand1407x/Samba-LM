import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax

# Memory fix must be first
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Import your existing classes from train.py
from train import MambaLM, TinyShakespeareLoader, create_train_state, train_step

def run_benchmark(n_layers, d_model, batch_size, block_size, num_steps=50):
    """Measures performance for a specific model configuration."""
    loader = TinyShakespeareLoader(block_size=block_size, batch_size=batch_size)
    
    key = jax.random.PRNGKey(42)
    model = MambaLM(vocab_size=loader.vocab_size, d_model=d_model, n_layers=n_layers)
    state = create_train_state(key, model, learning_rate=4e-4)
    
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"\nBenchmarking: {n_layers} Layers | {param_count:,} Parameters")

    # Warm-up step (JIT compilation)
    x, y = loader.get_batch('train')
    state, _ = train_step(state, x, y)
    
    # Timing loop
    start_time = time.time()
    for _ in range(num_steps):
        x, y = loader.get_batch('train')
        state, _ = train_step(state, x, y)
        jax.block_until_ready(state.params) # Ensure GPU finishes work
    
    end_time = time.time()
    
    total_time = end_time - start_time
    total_tokens = num_steps * batch_size * block_size
    tokens_per_sec = total_tokens / total_time
    
    return {
        'layers': n_layers,
        'params': param_count,
        'tokens_per_sec': tokens_per_sec,
        'time_per_step_ms': (total_time / num_steps) * 1000
    }

def main():
    configs = [
        {'layers': 4, 'd_model': 256}, # Your current "Safe Mode"
        {'layers': 8, 'd_model': 256}, # Doubled depth
    ]
    
    results = []
    print("="*60)
    print("PHASE 5: COMPARATIVE BENCHMARK (M2 PRO GPU)")
    print("="*60)

    for cfg in configs:
        res = run_benchmark(
            n_layers=cfg['layers'], 
            d_model=cfg['d_model'],
            batch_size=32, 
            block_size=128
        )
        results.append(res)
        print(f"Throughput: {res['tokens_per_sec']:.2f} tokens/sec")
        print(f"Latency: {res['time_per_step_ms']:.2f} ms/step")

    # Summary Table
    print("\n" + "-"*40)
    print(f"{'Layers':<10} | {'Tokens/Sec':<15} | {'Scaling'}")
    print("-"*40)
    base_speed = results[0]['tokens_per_sec']
    for r in results:
        scaling = (r['tokens_per_sec'] / base_speed) * 100
        print(f"{r['layers']:<10} | {r['tokens_per_sec']:<15.2f} | {scaling:.1f}%")

if __name__ == "__main__":
    main()