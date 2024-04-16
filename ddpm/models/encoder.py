import re
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from einops import rearrange
from functools import reduce
import numpy as np
from transformers import AutoTokenizer, AutoModel
from ddpm.models.unet_openai.attention import BasicTransformerBlock, CrossAttention


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        return inputs

    def encode(self, *args, **kwargs):
        return self(*args, **kwargs)


class FrozenBERTEmbedder(AbstractEncoder):
    use_text_split = False
    bert_max_length = 512
    def __init__(self, ckpt_path="/mnt/data/oss_beijing/dailinrui/data/pretrained/bert_chinese/", 
                 device="cuda", freeze=True, max_length=512):
        super().__init__()
        self.device = device
        self.max_length = max_length
        self.bert_max_length = 512
        assert self.max_length % self.bert_max_length == 0 or self.max_length < self.bert_max_length
        self.bert_encode_batch = self.max_length // self.bert_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path, local_files_only=True)
        self.transformer = AutoModel.from_pretrained(ckpt_path, local_files_only=True).to(self.device)
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False
            
    @staticmethod
    def token_split(string, max_length=bert_max_length):
        if len(string) < max_length:
            return [string]
        split_pos = [0] + [m.start() for m in re.finditer(r"\\\\|{", string)] + [len(string)]
        split_text = [string[split_pos[i]: split_pos[i+1]] for i in range(len(split_pos)-1)]
        
        def huffman_grouping(*t):
            if len(t) == 1:
                return t
            pair_len = [len(t[_] + t[_+1]) for _ in range(len(t)-1)]
            if min(pair_len) > max_length:
                return t
            pair_idx = np.argmin(pair_len)
            pair_t = t[pair_idx] + t[pair_idx + 1]
            if pair_idx + 2 < len(t):
                return huffman_grouping(*t[:pair_idx], pair_t, *t[pair_idx+2:])
            return huffman_grouping(*t[:pair_idx], pair_t)
            
        result_ls = huffman_grouping(*split_text)
        
        if max([len(_) for _ in result_ls]) > max_length:  # sep by "。"
            split_pos = [0] + [m.start() for m in re.finditer(r"。", string)] + [len(string)]
            split_text = [string[split_pos[i]: split_pos[i+1]] for i in range(len(split_pos)-1)]
            result_ls = huffman_grouping(*split_text)
            
        return result_ls
            
    def _merge_text_list(self, *ls):
        ls_ = []
        for l in ls:
            ls_.append(l)
            if not isinstance(l, list):
                assert isinstance(l:= str(l), str), f"got type {type(l)} for {l}, attempted conversion to str failed"
                ls_[-1] = self.token_split(l)
            if len(ls_[-1]) < self.bert_encode_batch:
                ls_[-1].append("")
            if len(ls_[-1]) > self.bert_encode_batch:
                ls_[-1] = l[:self.bert_encode_batch]
        return reduce(lambda x, y: x + y, ls_, [])
    
    def forward(self, text):
        if isinstance(text, str): text = [text]
        b = len(text)
        if self.use_text_split:
            text = self._merge_text_list(*text)
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.bert_max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        mask = batch_encoding["attention_mask"].to(self.device)
        outputs = self.transformer(input_ids=tokens, attention_mask=mask, return_dict=True)

        z = outputs.last_hidden_state
        z = rearrange(z, "(b x) n l -> b (n x) l", b=b, x=self.bert_encode_batch, n=self.bert_max_length)
        return z

    def encode(self, text):
        return self(text)
    
    
class PreloadedBERTEncoder(nn.Module):
    def __init__(self, embed_dim=768,
                 n_heads=8, depth=4, d_head=64, dropout=.1):
        super().__init__()
        self.embed_dim = embed_dim
        # self.proj_in = nn.Conv1d(embed_dim, d_head * n_heads, 1)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(embed_dim, n_heads, d_head, dropout=dropout)
                for d in range(depth)]
        )
        # self.proj_out = nn.Conv1d(n_heads * d_head, embed_dim, 1)
        
    def forward(self, inputs):
        # inputs: encoded text features of size [b embed_dim length]
        # outputs = self.proj_in(inputs)
        outputs = rearrange(inputs, "b c l -> b l c")
        for block in self.transformer_blocks:
            outputs = block(outputs)
        outputs = rearrange(outputs, "b l c -> b c l")
        # outputs = self.proj_out(outputs)
        return inputs + outputs
        