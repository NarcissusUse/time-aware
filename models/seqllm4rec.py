import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, OPTForCausalLM, AutoModelForCausalLM
from peft import (
    prepare_model_for_kbit_training,
)
class llm4rec(nn.Module):
    def __init__(
        self,
        device,
        llm_model="",
        max_output_txt_len=256,
        args= None
    ):
        super().__init__()
        self.device = device
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        self.args = args

        
        if llm_model == 'llama':
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        elif llm_model == 'llama-3b':
            model_id="meta-llama/Llama-3.2-3B-Instruct"
        else:
            raise Exception(f'{llm_model} is not supported')
        print()
        print("=========")

        if self.args.nn_parameter:
            self.llm_model = AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device, torch_dtype=torch.float16)
        else:
            self.llm_model = AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device, torch_dtype=torch.float16,load_in_8bit=True,)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
            
        
            
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': ['[UserRep]','[HistoryEmb]', '[UserOut]', '[ItemOut]']})
        self.llm_tokenizer.add_special_tokens({'cls_token': "[CLS]"})
        
        
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        self.llm_model = prepare_model_for_kbit_training(self.llm_model)
        
        for _, param in self.llm_model.named_parameters():
            if args.token:
                if 'token' in _:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False

        if not args.token:
            if args.nn_parameter:
                self.CLS = nn.Parameter(torch.normal(0,1, size = (1,self.llm_model.config.hidden_size))).to(device)
                self.CLS_item = nn.Parameter(torch.normal(0,1, size = (1,self.llm_model.config.hidden_size))).to(device)
            else:
                self.CLS = nn.Embedding(1,self.llm_model.config.hidden_size).to(device)
                nn.init.normal_(self.CLS.weight, mean = self.llm_model.model.embed_tokens.weight.mean(), std = self.llm_model.model.embed_tokens.weight.std())
                self.CLS_item = nn.Embedding(1,self.llm_model.config.hidden_size).to(device)
                nn.init.normal_(self.CLS_item.weight, mean = self.llm_model.model.embed_tokens.weight.mean(), std = self.llm_model.model.embed_tokens.weight.std())
        
        self.pred_user = nn.Sequential(
                nn.Linear(self.llm_model.config.hidden_size, 2048),
                nn.LayerNorm(2048),
                nn.LeakyReLU(),
                nn.Linear(2048, 128)
            )
        nn.init.xavier_normal_(self.pred_user[0].weight)
        nn.init.xavier_normal_(self.pred_user[3].weight)
        
        
        self.pred_item = nn.Sequential(
                nn.Linear(self.llm_model.config.hidden_size, 2048),
                nn.LayerNorm(2048),
                nn.LeakyReLU(),
                nn.Linear(2048, 128)
            )
        nn.init.xavier_normal_(self.pred_item[0].weight)
        nn.init.xavier_normal_(self.pred_item[3].weight)
        
        
        self.pred_user_CF2 = nn.Sequential(
                nn.Linear(64, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Linear(128, 128)
            )
        nn.init.xavier_normal_(self.pred_user_CF2[0].weight)
        nn.init.xavier_normal_(self.pred_user_CF2[3].weight)
        
        self.cf_to_latent2 = nn.Sequential(
                nn.Linear(64, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Linear(128, 128)
            )
        nn.init.xavier_normal_(self.cf_to_latent2[0].weight)
        nn.init.xavier_normal_(self.cf_to_latent2[3].weight)
        
        

        self.mse = nn.MSELoss()
        
        self.max_output_txt_len = max_output_txt_len
        
    
    def info_nce_loss_batch(self,anchor, log_emb, temperature=0.07):
        
        batch_size = anchor.shape[0]
        
        anchor = F.normalize(anchor, p=2, dim=1)#1
        log_emb = F.normalize(log_emb, p=2, dim=1)#1
        
        similarity_matrix = torch.matmul(anchor, log_emb.T)/temperature

        mask = torch.eye(batch_size, device= anchor.device).bool()
        
        pos_sim = similarity_matrix[mask].view(batch_size,1)
        neg_sim = similarity_matrix[~mask].view(batch_size, -1)
        
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        
        labels = torch.zeros(batch_size, dtype=torch.long, device = anchor.device)
        
                
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def rec_loss(self,anchor, items):
        
        
        logits = torch.bmm(items.view(anchor.shape[0], -1, anchor.shape[1]), anchor.unsqueeze(2)).squeeze(2)
        
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    
    def uniformity(self, x, p=2):
        return torch.pdist(x, p=p).pow(2).mul(-p).exp().mean()
    
    
    def replace_out_token_all(self, llm_tokens, inputs_embeds, token = [], embs= None,):
        for t in token:
            token_id = self.llm_tokenizer(t, return_tensors="pt", add_special_tokens=False).input_ids.item()
            vectors = []
            for inx in range(len(llm_tokens["input_ids"])):
                idx_tensor=(llm_tokens["input_ids"][inx]==token_id).nonzero().view(-1)
                user_vector = inputs_embeds[inx]
                if 'Emb' in t:
                    ee = embs[t][inx]
                    for idx, item_emb in zip(idx_tensor, ee):
                        user_vector = torch.cat((user_vector[:idx], item_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)
                elif 'Rep' in t:
                    for idx in idx_tensor:
                        user_emb = embs[t][inx]
                        user_vector = torch.cat((user_vector[:idx], user_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)
                else:
                    if not self.args.token:
                        for idx in idx_tensor:
                            if 'UserOut' in t:
                                if self.args.nn_parameter:
                                    user_vector = torch.cat((user_vector [:idx], self.CLS[torch.tensor([0]).to(self.device)], user_vector [idx+1:]), dim=0)
                                else:
                                    user_vector = torch.cat((user_vector [:idx], self.CLS(torch.tensor([0]).to(self.device)), user_vector [idx+1:]), dim=0)
                            elif 'ItemOut' in t:
                                if self.args.nn_parameter:
                                    user_vector = torch.cat((user_vector [:idx], self.CLS_item[torch.tensor([0]).to(self.device)], user_vector [idx+1:]), dim=0)
                                else:
                                    user_vector = torch.cat((user_vector [:idx], self.CLS_item(torch.tensor([0]).to(self.device)), user_vector [idx+1:]), dim=0)
            
                vectors.append(user_vector.unsqueeze(0))
            inputs_embeds = torch.cat(vectors)        
        return inputs_embeds
    
    def replace_out_token_all_infer(self, llm_tokens, inputs_embeds, token = [], embs= None, user_act = False, item_act = False):
        for t in token:
            token_id = self.llm_tokenizer(t, return_tensors="pt", add_special_tokens=False).input_ids.item()
            vectors = []
            for inx in range(len(llm_tokens["input_ids"])):
                idx_tensor=(llm_tokens["input_ids"][inx]==token_id).nonzero().view(-1)
                user_vector = inputs_embeds[inx]
                if 'Emb' in t:
                    ee = [embs[t][inx]]
                    # ee = embs[t][inx]
                    for idx, item_emb in zip(idx_tensor, ee):
                        user_vector = torch.cat((user_vector[:idx], item_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)
                                                
                elif 'Rep' in t:
                    for idx in idx_tensor:                        
                        user_emb = embs[t][inx]
                        user_vector = torch.cat((user_vector[:idx], user_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)
                else:
                    if not self.args.token:
                        for idx in idx_tensor:
                            if 'UserOut' in t:
                                if self.args.nn_parameter:
                                    user_vector = torch.cat((user_vector [:idx], self.CLS[torch.tensor([0]).to(self.device)], user_vector [idx+1:]), dim=0)
                                else:
                                    user_vector = torch.cat((user_vector [:idx], self.CLS(torch.tensor([0]).to(self.device)), user_vector [idx+1:]), dim=0)
                            elif 'ItemOut' in t:
                                if self.args.nn_parameter:
                                    user_vector = torch.cat((user_vector [:idx], self.CLS_item[torch.tensor([0]).to(self.device)], user_vector [idx+1:]), dim=0)
                                else:
                                    user_vector = torch.cat((user_vector [:idx], self.CLS_item(torch.tensor([0]).to(self.device)), user_vector [idx+1:]), dim=0)
                        
                
                vectors.append(user_vector.unsqueeze(0))
            inputs_embeds = torch.cat(vectors)        
        return inputs_embeds
        
    def get_embeddings(self, llm_tokens, token):
        token_idx = []
        token_id = self.llm_tokenizer(token, return_tensors="pt", add_special_tokens=False).input_ids.item()
        for inx in range(len(llm_tokens['input_ids'])):
            idx_tensor = (llm_tokens['input_ids'][inx] == token_id).nonzero().view(-1)
            token_idx.append(idx_tensor)
        return token_idx


    
    def forward(self, samples, mode = 0):
        if mode ==0:
            return self.train_mode0(samples)
        elif mode == 1:
            return self.train_mode1(samples)

    def train_mode0(self,samples):
        max_input_length = 1024
        log_emb = samples['log_emb']
        llm_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_input_length,
        ).to(self.device)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        
        # no user
        inputs_embeds = self.replace_out_token_all(llm_tokens, inputs_embeds, token = ['[UserOut]', '[HistoryEmb]'], embs= { '[HistoryEmb]':samples['interact']})

        
        candi_tokens = self.llm_tokenizer(
                samples['candidates_pos'],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_input_length,
            ).to(self.device)

        candi_embeds = self.llm_model.get_input_embeddings()(candi_tokens['input_ids'])

        candi_embeds = self.replace_out_token_all_infer(candi_tokens, candi_embeds, token = ['[ItemOut]', '[HistoryEmb]'], embs= {'[HistoryEmb]':samples['candidate_embs']})

                
        with torch.amp.autocast('cuda'):
            
            candi_outputs = self.llm_model.forward(
                inputs_embeds=candi_embeds,
                output_hidden_states=True
            )
            
            indx = self.get_embeddings(candi_tokens, '[ItemOut]')
            item_outputs = torch.cat([candi_outputs.hidden_states[-1][i,indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])

            outputs = self.llm_model.forward(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True
            )
            
            indx = self.get_embeddings(llm_tokens, '[UserOut]')
            user_outputs = torch.cat([outputs.hidden_states[-1][i,indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])

        
        user_outputs = self.pred_user(user_outputs)
        item_outputs = self.pred_item(item_outputs)

        rec_loss = self.rec_loss(user_outputs, item_outputs)

        log_emb = self.pred_user_CF2(log_emb)


        user_outputs = F.normalize(user_outputs, p=2, dim=1)#1
        log_emb = F.normalize(log_emb, p=2, dim=1)#1

        match_loss = self.mse(user_outputs,log_emb)
        
        match_loss += (self.uniformity(user_outputs)+ self.uniformity(log_emb))

        
        loss = rec_loss + match_loss
        

        return loss, rec_loss.item(), match_loss.item()
    
    
    
    