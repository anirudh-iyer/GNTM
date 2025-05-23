# evaluate_best.py
import torch
from main import init_config, main  # Assuming main.py is modified as in our previous code

def evaluate_best_model():
    # Initialize configuration
    args = init_config()
    # Switch to evaluation mode
    args.eval = True
    # Optionally, you may override other parameters if needed.
    
    # Run the main function in evaluation mode.
    # The evaluation section of your main() will load the best saved model and print topics/plots.
    main(args)

if __name__ == '__main__':
    evaluate_best_model()
