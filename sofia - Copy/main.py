from smarter import input_data
import tensorflow as tf
import argparse
import numpy as np


# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_string('data_dir', 'data/', 'Directory to read the training data.')
# flags.DEFINE_string('sessions_dir', '/media/sf_shared/Tutorial/smarter/sessions/', 'Directory to save/read the trained sessions.')
# flags.DEFINE_string('logs_dir', 'logs/', 'Directory to read the training data.')

# python3 train.py -a train -m service :train model to predict the affected service
# python3 train.py -a train -m os      :train model to predict the os to create

def main(_):
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('-a', help='action train, infer or bulk_infer', choices=['train', 'infer', 'bulk_infer'])
    parser.add_argument('-m', help='model service, cause, os', choices=['service', 'cause', 'os'])
    parser.add_argument('-k', help='top k predictions')
    args = parser.parse_args()

    if args.a == 'train':
        from smarter import smarter_training as train
    else:
        from smarter import smarter_prediction as prediction

    if args.a == 'train':
        data_sets = input_data.read_data_sets('data/')
        train.run_training(args.m, data_sets)
    if args.a == 'bulk_infer':
        data_sets = input_data.read_data_sets('data/')
        if args.k == None:
            k = 1
        else:
            k = int(args.k)
        prediction.run_bulk_inference(args.m, data_sets, k=k)
    if args.a == 'infer':
        # Get the sets of features and labels test on SMART data.
        if args.m == 'service':
            data = "001001000010000000000000000000000000000000000000000010000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
            features = np.fromstring(data, dtype=np.int8) - 48
            features = features.reshape(1, len(data))
        elif args.m == 'os':
            data = "001001000010000000000000000000000000000000000000000010000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
            features = np.fromstring(data, dtype=np.int8) - 48
            features = features.reshape(1, len(data))
        if args.m == 'cause':
            data = "011000100001000000000000000000000000000000000000000100000000000000000000010101001001000010010000000000000100000000000010000000000000000000001000000001001001001000000000000000100010000001001000000000001010000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000100000000000000000000"
            features = np.fromstring(data, dtype=np.int8) - 48
            features = features.reshape(1, len(data))
        else:
            data = ""

        prd = prediction.run_inference(args.m, features)
        print("Predicted value: ", prd)


if __name__ == '__main__':
    tf.app.run()