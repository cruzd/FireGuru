=====
Smarter
=====

Smarter is a simple Tensorflow app that can be trained to predict the next 
step to take while handling customer support calls.

Detailed documentation is in the "docs" directory.

Quick start


1. Run 'python3 main.py -a train -m cause' to train the neural network for smart cause recommendation
with data from ./data/train-data.csv

2. Run 'python3 main.py -a bulk_infer -m cause' to test the neural network for smart recommendation
with data from /data/test-data.csv
