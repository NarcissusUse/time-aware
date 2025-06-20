# Code based on https://github.com/ghdtjr/A-LLMRec
import os
import sys
import argparse
import importlib
from utils import *
from train_model import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--multi_gpu", action='store_true')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--llm", type=str, default='llama-1b', help='flan_t5, llama, vicuna')
    parser.add_argument("--recsys", type=str, default='sasrec')
    parser.add_argument("--rec_pre_trained_data", type=str, default='Movies_and_TV')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--extract", action='store_true')
    parser.add_argument("--token", action='store_true')
    parser.add_argument("--save_dir", type=str, default='seqllm')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--batch_size_infer', default=4, type=int)
    
    # Add model version argument
    parser.add_argument("--model_version", type=str, default='', 
                       help='Model version suffix (B, C, D, E, F). Leave empty for base model')
    
    parser.add_argument('--infer_epoch', default=1, type=int)
    parser.add_argument('--maxlen', default=128, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument("--stage2_lr", type=float, default=0.0001)
    parser.add_argument('--nn_parameter', default=False, action='store_true')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'hpu':
        args.device = torch.device('hpu')
    else:
        args.device = 'cuda:' + str(args.device)
    
    # Pass the model version to the training function
    if args.train:
        train_model(args)
