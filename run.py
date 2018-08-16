import argparse

def main(_):
    print("Fire Guru - Run started")
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('-a', help='Which action? (Train or Prepare Data)', 
        choices=['train', 'data'])
    args = parser.parse_args()

    if args.a == 'train':
        from training import guru_train as train
        train.run_training()
    if args.a == 'data':
        from data_prep import process_data as data
        data.run_data_process()


