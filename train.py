"""
Training script for IMDER
dataset_name: Selecting dataset (mosi or mosei)
seeds: This is a list containing running seeds you input
mr: missing rate ranging from 0.1 to 0.7
"""
from run import IMDER_run

IMDER_run(model_name='imder',
           dataset_name='mosi',
           seeds=[1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119],
           mr=0.1)
