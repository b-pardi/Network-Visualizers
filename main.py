import argparse
import numpy as np
import src.xor as xor
import src.mnist_digits as mnist_digits

def main():
    parser = argparse.ArgumentParser(description="Choose network to visualize")
    script_group = parser.add_mutually_exclusive_group(required=True)

    # XOR classifier options
    script_group.add_argument('--xor', action='store_true', help='Run the XOR classifier')
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs to train (default 30,000)')
    parser.add_argument('-hn', '--hidden_neurons', type=int, help='Number of neurons in hidden layer (default 3)')
    parser.add_argument('-ner', '--num_epochs_refresh_visualizer', type=int, help='Number of epochs per refreshing of visualizers (effects speed of training, default 100)')

    # MNIST classifier options
    script_group.add_argument('-md', '--mnist_digits', action='store_true', help='Run MNIST classifier')
    parser.add_argument('-m', '--mode', choices=['train', 'infer'], default='train', help="Choose whether to train a CNN model (default) or infer on a pretrained one")
    parser.add_argument('-uc', '--user_config', action='store_true', help="Use user-defined CNN configuration file (config/user_cnn_params.json) for training CNN on MNIST")
    parser.add_argument('-um', '--user_model', type=str, help="File path to user's previously trained model object. (Default in the 'models/' folder) Note: Must have been trained with this specific classifier, as it relies on CNN object properties defined in the config file used for inference.")
    args = parser.parse_args()

    # XOR Neural Network argument parsing
    if args.xor:
        kwargs = {}
        if args.hidden_neurons:
            kwargs['h'] = args.hidden_neurons
        if args.num_epochs_refresh_visualizer:
            kwargs['ner'] = args.num_epochs_refresh_visualizer
        xor.run_xor(**kwargs)

    # CNN MNIST argument parsing
    elif args.mnist_digits:
        if args.mode == 'train': # run mnist training visuals
            if args.user_config: # if user opted to use custome cnn parameters
                config_fp = "config/user_cnn_params.json"
            else: # use default config parameters
                config_fp = "config/default_cnn_params.json"
            mnist_digits.run_mnist_train(config_fp)
        
        elif args.mode == 'infer': # run mnist inference visuals
            if args.user_model: # if user opted to use their model trained with this project
                model_fp = args.user_model
            else: # use existing default pretrained model trained with default configs
                model_fp = "models/default_mnist_cnn.pkl"
            mnist_digits.run_mnist_inference(model_fp)


if __name__ == '__main__':
    main()