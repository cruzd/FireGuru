import argparse
import tensorflow as tf

def main(_):
    print("Fire Guru - Run started")
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('-a', help='Which action? (Train or Prepare Data)', 
        choices=['train', 'data'])
    parser.add_argument('-m', help='Real data or pessimist?', 
        choices=['real', 'pessimist'])
    args = parser.parse_args()

    if args.a == 'train':
        if args.m == 'pessimist':
            file='pessimist'
        if args.m == 'real':
            file='real'
        from training import guru_train as train
        train.run_training(file)
    if args.a == 'data':
        if args.m == 'pessimist':
            from data_prep import pessimist_data as data
        if args.m == 'real':
            from data_prep import real_data as data
        data.run_data_process()

if __name__ == '__main__':
    tf.app.run()


