import argparse
import numpy as np
import src.xor as xor
import src.mnist_digits as mnist_digits

def main():
    parser = argparse.ArgumentParser(description="Choose network to visualize")
    script_group = parser.add_mutually_exclusive_group(required=True)

    script_group.add_argument('--xor', action='store_true', help='Run the XOR classifier')
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs to train (default 30,000)')
    parser.add_argument('-hn', '--hidden_neurons', type=int, help='Number of neurons in hidden layer (default 3)')
    parser.add_argument('-ner', '--num_epochs_refresh_visualizer', type=int, help='Number of epochs per refreshing of visualizers (effects speed of training, default 100)')

    script_group.add_argument('-md', '--mnist_digits', action='store_true', help='Run MNIST classifier')
    

    args = parser.parse_args()
    if args.xor:
        kwargs = {}
        if args.hidden_neurons:
            kwargs['h'] = args.hidden_neurons
        if args.num_epochs_refresh_visualizer:
            kwargs['ner'] = args.num_epochs_refresh_visualizer
        xor.run_xor(**kwargs)

    elif args.mnist_digits:
        mnist_digits.run_mnist()


if __name__ == '__main__':
    main()