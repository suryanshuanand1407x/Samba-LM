import json
import matplotlib.pyplot as plt

def plot_training_results():
    # 1. Load the data
    try:
        with open('training_history.json', 'r') as f:
            history = json.load(f)
    except FileNotFoundError:
        print("Error: 'training_history.json' not found.")
        print("Please run 'train.py' first to generate the data.")
        return

    steps = history['step']
    train_loss = history['train_loss']
    val_loss = history['val_loss']

    # 2. Create the Plot
    plt.figure(figsize=(10, 6))
    
    # Plot Training Loss (Blue)
    plt.plot(steps, train_loss, label='Training Loss', marker='o', 
             linestyle='-', color='#1f77b4', linewidth=2)
    
    # Plot Validation Loss (Orange)
    plt.plot(steps, val_loss, label='Validation Loss', marker='s', 
             linestyle='--', color='#ff7f0e', linewidth=2)

    # 3. Styling
    plt.title('Mamba Training Progress: Train vs Validation', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Cross Entropy Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # Annotate the final Validation Loss
    final_val = val_loss[-1]
    final_step = steps[-1]
    plt.annotate(f'Final Val: {final_val:.4f}', 
                 xy=(final_step, final_val), 
                 xytext=(final_step - 500, final_val + 0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    # 4. Save and Show
    output_file = 'loss_graph.png'
    plt.savefig(output_file, dpi=300)
    print(f"Graph saved as '{output_file}'")
    plt.show()

if __name__ == "__main__":
    plot_training_results()