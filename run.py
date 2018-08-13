def main(_):
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('-a', help='Which action? (Train or Prepare Data)', 
        choices=['train', 'data'])
    args = parser.parse_args()

    if args.a == 'train':
        from training import guru_train as train
        from training import guru_model as model
        session = model.set_model()
        train.run_training(session)
    if args.a == 'data':
        from data_prep import process_data as data
        data.run_data_process()

if __name__ == '__main__':
    tf.app.run()