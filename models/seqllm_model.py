import random
import pickle

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import numpy as np

from models.recsys_model import *
from models.seqllm4rec import *
from sentence_transformers import SentenceTransformer
from datetime import datetime

from tqdm import trange, tqdm

try:
    import habana_frameworks.torch.core as htcore
except:
    0
    

class llmrec_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        rec_pre_trained_data = args.rec_pre_trained_data
        self.args = args
        self.device = args.device
        
        with open(f'./SeqRec/data_{args.rec_pre_trained_data}/text_name_dict.json.gz','rb') as ft:
            self.text_name_dict = pickle.load(ft)
        
        self.recsys = RecSys(args.recsys, rec_pre_trained_data, self.device)

        self.item_num = self.recsys.item_num
        self.rec_sys_dim = self.recsys.hidden_units
        self.sbert_dim = 768

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.all_embs = None
        self.maxlen = args.maxlen
        self.NDCG = 0
        self.HIT = 0
        self.NDCG_20 = 0
        self.HIT_20 = 0
        
        
        self.rec_NDCG = 0
        self.rec_HIT = 0
        self.lan_NDCG=0
        self.lan_HIT=0
        self.num_user = 0
        self.yes = 0

        self.extract_embs_list = []
        
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
            
        self.llm = llm4rec(device=self.device, llm_model=args.llm, args = self.args)

        self.item_emb_proj = nn.Sequential(
            nn.Linear(self.rec_sys_dim, self.llm.llm_model.config.hidden_size),
            nn.LayerNorm(self.llm.llm_model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llm.llm_model.config.hidden_size, self.llm.llm_model.config.hidden_size)
        )
        nn.init.xavier_normal_(self.item_emb_proj[0].weight)
        nn.init.xavier_normal_(self.item_emb_proj[3].weight)
        
        self.users = 0.0
        self.NDCG = 0.0
        self.HT = 0.0
            

            
    def save_model(self, args, epoch2=None, best=False):
        out_dir = f'./models/{args.save_dir}/'
        if best:
            out_dir = out_dir[:-1] + 'best/'
        
        create_dir(out_dir)
        out_dir += f'{args.rec_pre_trained_data}_'
        
        out_dir += f'{args.llm}_{epoch2}_'
        if args.train:
            torch.save(self.item_emb_proj.state_dict(), out_dir + 'item_proj.pt')
            torch.save(self.llm.pred_user.state_dict(), out_dir + 'pred_user.pt')
            torch.save(self.llm.pred_item.state_dict(), out_dir + 'pred_item.pt')
            if not args.token:
                if args.nn_parameter:
                    torch.save(self.llm.CLS.data, out_dir + 'CLS.pt')
                    torch.save(self.llm.CLS_item.data, out_dir + 'CLS_item.pt')
                else:
                    torch.save(self.llm.CLS.state_dict(), out_dir + 'CLS.pt')
                    torch.save(self.llm.CLS_item.state_dict(), out_dir + 'CLS_item.pt')
            if args.token:
                torch.save(self.llm.llm_model.model.embed_tokens.state_dict(), out_dir + 'token.pt')
  
            
    def load_model(self, args, phase1_epoch=None, phase2_epoch=None):
        out_dir = f'./models/{args.save_dir}/{args.rec_pre_trained_data}_'

        out_dir += f'{args.llm}_{phase2_epoch}_'
        
        
        item_emb_proj = torch.load(out_dir + 'item_proj.pt', map_location = self.device)
        self.item_emb_proj.load_state_dict(item_emb_proj)
        del item_emb_proj
        
        
        pred_user = torch.load(out_dir + 'pred_user.pt', map_location = self.device)
        self.llm.pred_user.load_state_dict(pred_user)
        del pred_user
        
        pred_item = torch.load(out_dir + 'pred_item.pt', map_location = self.device)
        self.llm.pred_item.load_state_dict(pred_item)
        del pred_item
        
        if not args.token:
            CLS = torch.load(out_dir + 'CLS.pt', map_location = self.device)
            self.llm.CLS.load_state_dict(CLS)
            del CLS
            
            CLS_item = torch.load(out_dir + 'CLS_item.pt', map_location = self.device)
            self.llm.CLS_item.load_state_dict(CLS_item)
            del CLS_item
        
        if args.token:
            token = torch.load(out_dir + 'token.pt', map_location = self.device)
            self.llm.llm_model.model.embed_tokens.load_state_dict(token)
            del token
            

    def find_item_text(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'

        if title_flag and description_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}, {self.text_name_dict[d].get(i,d_)}"' for i in item]
        elif title_flag and not description_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}"' for i in item]
        elif not title_flag and description_flag:
            return [f'"{self.text_name_dict[d].get(i,d_)}"' for i in item]
        
    def find_item_time(self, item, user, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'

        l = [datetime.utcfromtimestamp(int(self.text_name_dict['time'][i][user])/1000) for i in item]
        return [l_.strftime('%Y-%m-%d') for l_ in l]
    

    def find_item_text_single(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'
        
        if title_flag and description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}, {self.text_name_dict[d].get(item,d_)}"'
        elif title_flag and not description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}"'
        elif not title_flag and description_flag:
            return f'"{self.text_name_dict[d].get(item,d_)}"'
        
    def get_item_emb(self, item_ids):
        with torch.no_grad():
            if self.args.nn_parameter:
                item_embs = self.recsys.model.item_emb[torch.LongTensor(item_ids).to(self.device)]
            else:
                item_embs = self.recsys.model.item_emb(torch.LongTensor(item_ids).to(self.device))
        
        return item_embs
    
    def forward(self, data, optimizer=None, batch_iter=None, mode='phase1'):
        if mode == 'phase2':
            self.pre_train_phase2(data, optimizer, batch_iter)
        if mode=='generate_batch':
            self.generate_batch(data)
            print(self.args.save_dir, self.args.rec_pre_trained_data)
            print('test (NDCG@10: %.4f, HR@10: %.4f), Num User: %.4f'
                    % (self.NDCG/self.users, self.HT/self.users, self.users))
            print('test (NDCG@20: %.4f, HR@20: %.4f), Num User: %.4f'
                    % (self.NDCG_20/self.users, self.HIT_20/self.users, self.users))
        if mode=='extract':
            self.extract_emb(data)

    def make_interact_text(self, interact_ids, interact_max_num, user):
        interact_item_titles_ = self.find_item_text(interact_ids, title_flag=True, description_flag=False)
        times = self.find_item_time(interact_ids, user)
        interact_text = []
        count = 1
        
            
        if interact_max_num =='all':
            times = self.find_item_time(interact_ids, user)
        else:
            times = self.find_item_time(interact_ids[-interact_max_num:], user)
        
        if interact_max_num == 'all':
            for title in interact_item_titles_:
                interact_text.append(f'Item No.{count}, Time: {times[count-1]}, ' + title + '[HistoryEmb]')

                count+=1
        else:
            for title in interact_item_titles_[-interact_max_num:]:
                interact_text.append(f'Item No.{count}, Time: {times[count-1]}, ' + title + '[HistoryEmb]')
                
                count+=1
            interact_ids = interact_ids[-interact_max_num:]
            
        interact_text = ','.join(interact_text)
        return interact_text, interact_ids

    
    
    def make_candidate_text(self, interact_ids, candidate_num, target_item_id, target_item_title, candi_set = None, task = 'ItemTask'):
        neg_item_id = []
        if candi_set == None:
            neg_item_id = []
            while len(neg_item_id)<99:
                t = np.random.randint(1, self.item_num+1)
                if not (t in interact_ids or t in neg_item_id):
                    neg_item_id.append(t)
        else:
            his = set(interact_ids)
            items = list(candi_set.difference(his))
            if len(items) >99:
                neg_item_id = random.sample(items, 99)
            else:
                while len(neg_item_id)<49:
                    t = np.random.randint(1, self.item_num+1)
                    if not (t in interact_ids or t in neg_item_id):
                        neg_item_id.append(t)
        random.shuffle(neg_item_id)
        
        candidate_ids = [target_item_id]
        
        candidate_text = [f'The item title and item embedding are as follows: ' + target_item_title + "[HistoryEmb], then generate item representation token:[ItemOut]"]


        for neg_candidate in neg_item_id[:candidate_num - 1]:
            candidate_text.append(f'The item title and item embedding are as follows: ' + self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False) + "[HistoryEmb], then generate item representation token:[ItemOut]")

            candidate_ids.append(neg_candidate)
            
        return candidate_text, candidate_ids
    
    
    def make_candidate(self, interact_ids, candidate_num, target_item_id, target_item_title, candi_set = None, task = 'ItemTask'):
        neg_item_id = []
        neg_item_id = []
        while len(neg_item_id)<99:
            t = np.random.randint(1, self.item_num+1)
            if not (t in interact_ids or t in neg_item_id):
                neg_item_id.append(t)
        
        random.shuffle(neg_item_id)
        
        candidate_ids = [target_item_id]
        
        candidate_ids = candidate_ids + neg_item_id[:candidate_num - 1]
            
        return candidate_ids
    
    
    def pre_train_phase2(self, data, optimizer, batch_iter):
        epoch, total_epoch, step, total_step = batch_iter
        print(self.args.save_dir, self.args.rec_pre_trained_data, self.args.llm)
        optimizer.zero_grad()
        u, seq, pos, neg = data
        
        original_seq = seq.copy()
        
        
        mean_loss = 0
        
        text_input = []
        candidates_pos = []
        candidates_neg = []
        interact_embs = []
        candidate_embs_pos = []
        candidate_embs_neg = []
        candidate_embs = []
        
        loss_rm_mode1 = 0
        loss_rm_mode2 = 0
        
        with torch.no_grad():
            log_emb = self.recsys.model(u,seq,pos,neg, mode = 'log_only')
        
        for i in range(len(u)):
            target_item_id = pos[i][-1]
            target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
            
            

            interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10, u[i])
            candidate_num = 4
            candidate_text, candidate_ids = self.make_candidate_text(seq[i][seq[i]>0], candidate_num, target_item_id, target_item_title, task='RecTask')

                #no user
            input_text = ''
                

            input_text += 'This user has made a series of purchases in the following order: '
                
            input_text += interact_text
            
            input_text +=". Based on this sequence of purchases, generate user representation token:[UserOut]"

            text_input.append(input_text)
            
            candidates_pos += candidate_text             
            
            
            interact_embs.append(self.item_emb_proj((self.get_item_emb(interact_ids))))
            candidate_embs_pos.append(self.item_emb_proj((self.get_item_emb([candidate_ids]))).squeeze(0))
                        
        
        candidate_embs = torch.cat(candidate_embs_pos)
        
        
        samples = {'text_input': text_input, 'log_emb':log_emb, 'candidates_pos': candidates_pos, 'interact': interact_embs, 'candidate_embs':candidate_embs,}
        
        loss, rec_loss, match_loss = self.llm(samples, mode=0)
                    
        print("LLMRec model loss in epoch {}/{} iteration {}/{}: {}".format(epoch, total_epoch, step, total_step, rec_loss))
                            
        print("LLMRec model Matching loss in epoch {}/{} iteration {}/{}: {}".format(epoch, total_epoch, step, total_step, match_loss))

        loss.backward()
        if self.args.nn_parameter:
            htcore.mark_step()
        optimizer.step()
        if self.args.nn_parameter:
            htcore.mark_step()
        
    
    def split_into_batches(self,itemnum, m):
        numbers = list(range(1, itemnum+1))
        
        batches = [numbers[i:i + m] for i in range(0, itemnum, m)]
        
        return batches

    
    def generate_batch(self,data):
        if self.all_embs == None:
            batch_ = 128
            if self.args.llm =='llama':
                batch_ = 64
            if self.args.rec_pre_trained_data == 'Electronics' or self.args.rec_pre_trained_data == 'Books':
                batch_ = 64
                if self.args.llm =='llama':
                    batch_ = 32
            batches = self.split_into_batches(self.item_num, batch_)#128
            self.all_embs = []
            max_input_length = 1024
            for bat in tqdm(batches):
                candidate_text = []
                candidate_ids = []
                candidate_embs = []
                for neg_candidate in bat:
                    candidate_text.append('The item title and item embedding are as follows: ' + self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False) + "[HistoryEmb], then generate item representation token:[ItemOut]")
                    
                    candidate_ids.append(neg_candidate)
                with torch.no_grad():
                    candi_tokens = self.llm.llm_tokenizer(
                        candidate_text,
                        return_tensors="pt",
                        padding="longest",
                        truncation=True,
                        max_length=max_input_length,
                    ).to(self.device)
                    candidate_embs.append(self.item_emb_proj((self.get_item_emb(candidate_ids))))

                    candi_embeds = self.llm.llm_model.get_input_embeddings()(candi_tokens['input_ids'])
                    candi_embeds = self.llm.replace_out_token_all_infer(candi_tokens, candi_embeds, token = ['[ItemOut]', '[HistoryEmb]'], embs= {'[HistoryEmb]':candidate_embs[0]})
                    
                    with torch.amp.autocast('cuda'):
                        candi_outputs = self.llm.llm_model.forward(
                            inputs_embeds=candi_embeds,
                            output_hidden_states=True
                        )
                        
                        indx = self.llm.get_embeddings(candi_tokens, '[ItemOut]')
                        item_outputs = torch.cat([candi_outputs.hidden_states[-1][i,indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])
                        
                        item_outputs = self.llm.pred_item(item_outputs)
                    
                    self.all_embs.append(item_outputs)
                    del candi_outputs
                    del item_outputs        
            self.all_embs = torch.cat(self.all_embs)
            
        u, seq, pos, neg, rank, candi_set, files = data
        original_seq = seq.copy()
        
        text_input = []
        interact_embs = []
        candidate = []
        with torch.no_grad():
            for i in range(len(u)):

                candidate_embs = []
                target_item_id = pos[i]
                target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
                
                interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10, u[i])
                

                candidate_num = 100
                candidate_ids = self.make_candidate(seq[i][seq[i]>0], candidate_num, target_item_id, target_item_title, candi_set)
                
                candidate.append(candidate_ids)
                

                #no user
                input_text = ''
                    

                input_text += 'This user has made a series of purchases in the following order: '
                    
                input_text += interact_text
                

                input_text +=". Based on this sequence of purchases, generate user representation token:[UserOut]"
                
                text_input.append(input_text)
                
                
                interact_embs.append(self.item_emb_proj((self.get_item_emb(interact_ids))))
                

            max_input_length = 1024
            
            llm_tokens = self.llm.llm_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_input_length,
            ).to(self.device)
            
            inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
            
                #no user
            inputs_embeds = self.llm.replace_out_token_all(llm_tokens, inputs_embeds, token = ['[UserOut]', '[HistoryEmb]'], embs= { '[HistoryEmb]':interact_embs})

            with torch.cuda.amp.autocast():
                outputs = self.llm.llm_model.forward(
                    inputs_embeds=inputs_embeds,

                    output_hidden_states=True
                )
                
                indx = self.llm.get_embeddings(llm_tokens, '[UserOut]')
                user_outputs = torch.cat([outputs.hidden_states[-1][i,indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])
                user_outputs = self.llm.pred_user(user_outputs)
                
                for i in range(len(candidate)):
                    
                    item_outputs = self.all_embs[np.array(candidate[i])-1]
                    
                    logits= torch.mm(item_outputs, user_outputs[i].unsqueeze(0).T).squeeze(-1)
                
                    logits = -1*logits
                    
                    rank = logits.argsort().argsort()[0].item()
                    
                    if rank < 10:
                        self.NDCG += 1 / np.log2(rank + 2)
                        self.HT += 1
                    if rank < 20:
                        self.NDCG_20 += 1 / np.log2(rank + 2)
                        self.HIT_20 += 1
                    self.users +=1
        return self.NDCG
                
    def extract_emb(self,data):    
        u, seq, pos, neg, original_seq, rank, files = data
            
        text_input = []
        interact_embs = []
        candidate = []
        with torch.no_grad():
            for i in range(len(u)):

                interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10, u[i])

                input_text = ''
                    

                input_text += 'This user has made a series of purchases in the following order: '
                    
                input_text += interact_text
                

                input_text +=". Based on this sequence of purchases, generate user representation token:[UserOut]"
                
                text_input.append(input_text)
                
                interact_embs.append(self.item_emb_proj((self.get_item_emb(interact_ids))))
                

            max_input_length = 1024
            
            llm_tokens = self.llm.llm_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_input_length,
            ).to(self.device)
            
            inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
            
            inputs_embeds = self.llm.replace_out_token_all(llm_tokens, inputs_embeds, token = ['[UserOut]', '[HistoryEmb]'], embs= { '[HistoryEmb]':interact_embs})

            with torch.cuda.amp.autocast():
                outputs = self.llm.llm_model.forward(
                    inputs_embeds=inputs_embeds,

                    output_hidden_states=True
                )
                
                indx = self.llm.get_embeddings(llm_tokens, '[UserOut]')
                user_outputs = torch.cat([outputs.hidden_states[-1][i,indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])
                user_outputs = self.llm.pred_user(user_outputs)
                
                self.extract_embs_list.append(user_outputs.detach().cpu())
                
        return 0
