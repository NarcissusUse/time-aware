import os
import time
import torch
import argparse
import numpy as np
import sys

from model import SASRec
from data_preprocess import *
from utils import *

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=128, type=int)
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.1, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='0', type=str, help='cpu, hpu, gpu -> num')

parser.add_argument('--inference_only', default=False, action='store_true')
parser.add_argument('--nn_parameter', default=False, action='store_true')
parser.add_argument('--state_dict_path', default=None, type=str)

args = parser.parse_args()

if __name__ == '__main__':
    
    # global dataset
    if args.device =='hpu':
        args.is_hpu = True
    else:
        args.is_hpu = False
        
    if (not os.path.isfile(f'./../data_{args.dataset}/{args.dataset}_train.txt')) or (not os.path.isfile(f'./../data_{args.dataset}/{args.dataset}_valid.txt') or (not os.path.isfile(f'./../data_{args.dataset}/{args.dataset}_test.txt'))):
        print("Download Dataset")
        if not os.path.exists(f'./../data_{args.dataset}'):
            os.makedirs(f'./../data_{args.dataset}')
        preprocess_raw_5core(args.dataset)
    dataset = data_partition(args.dataset, args)
    
    
    [user_train, user_valid, user_test, usernum, itemnum, eval_set] = dataset
    print('user num:', usernum, 'item num:', itemnum)
    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    if args.device =='hpu':
        ###GAUDI
        import habana_frameworks.torch.core as htcore
        args.device = torch.device('hpu')
        
        # IF nn.Embedding Error solve in Gaudi, then remove this command
        args.nn_parameter = True
    elif args.device != 'hpu' and args.device != 'cpu':
        args.device = 'cuda:'+str(args.device)
    
    # dataloader
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)       
    # model init
    model = SASRec(usernum, itemnum, args).to(args.device)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            kwargs, checkpoint = torch.load(args.state_dict_path)
            kwargs['args'].device = args.device
            model = SASRec(**kwargs).to(args.device)
            model.load_state_dict(checkpoint)
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
    
    if args.inference_only:
        # save_eval(model, dataset, args)

        print('Evaluate')
        
        
        t_test = evaluate(model, dataset, args, ranking = 10)
        print('')
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
                
        t_test = evaluate(model, dataset, args, ranking = 20)
        print('')
        print('test (NDCG@20: %.4f, HR@20: %.4f)' % (t_test[0], t_test[1]))
        
                
        sys.exit("Terminating Inference")
        
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    time_list = []
    loss_list = []
    T = 0.0
    t0 = time.time()
    start_time = time.time()
    
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        model.train()
        epoch_s_time = time.time()
        total_loss, count = 0, 0
        if args.inference_only: break
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)

            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            
            #nn.Embedding
            if args.nn_parameter:
                loss += args.l2_emb * torch.norm(model.item_emb)
            else:
                for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
             
            #GAUDI
            loss.backward()
            if args.is_hpu:
                htcore.mark_step()
            adam_optimizer.step()
            if args.is_hpu:
                htcore.mark_step()
            
            total_loss += loss.item()
            count+=1
            
            if step % 100 == 0:
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))
        
        epoch_e_time = time.time()
        time_list.append(epoch_e_time - epoch_s_time)
        loss_list.append(total_loss/count)
    
        if epoch == args.num_epochs:
            folder = args.dataset
            fname = 'SASRec_saving.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            if not os.path.exists(os.path.join(folder, fname)):
                try:
                    os.makedirs(os.path.join(folder))
                except:
                    print()
            torch.save([model.kwargs, model.state_dict()], os.path.join(folder, fname))
    
    sampler.close()
    end_time = time.time()
    
    save_eval(model, dataset, args)
    
    print("Done")
    print("Time:", end_time-start_time)
