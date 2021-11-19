from os import POSIX_FADV_SEQUENTIAL, replace
import torch
import json
import os
import random
from tqdm import tqdm
from torch._C import dtype
from torch.utils.data import Dataset
from data_utils.tokenization_enc_dec import EncDecTokenizer
import pickle
import mpu
import math
from utils import print_rank_0, save_rank_0

class T5Dataset(Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None, lang=None):
        self.args = args
        self.tokenizer = tokenizer
        self.ratio = ratio
        self.path = path
        self.pad_id = tokenizer.pad_id
        self.prefix = prefix
        self.prefix_ids = self.tokenizer.encode(prefix) if prefix is not None else []
        self.enc_seq_length = args.enc_seq_length - len(self.prefix_ids)
        self.add_target_post=add_target_post
        self.split = split
        self.do_infer = do_infer
        self.idx = 0
        self.prompt_config = prompt_config
        self.prompt_len = self.prompt_config["enc"]["prompt_len"]
        self.lang = lang

        if cache_path is not None:
            cache_path = os.path.join(cache_path, "cache_{}_{}.pkl".format(path.replace("/", "_"), ratio))
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    self.data, self.max_enc_len, self.max_dec_len = pickle.load(f)
            else:
                self.data, self.max_enc_len, self.max_dec_len = self.process_data()
                with open(cache_path, "wb") as f:
                    pickle.dump((self.data, self.max_enc_len, self.max_dec_len), f)
        else:
            self.data, self.max_enc_len, self.max_dec_len = self.process_data()

        if prompt_config is not None:
            self.data, self.max_enc_len, self.max_dec_len = self.add_prompt_ids(self.data, self.max_enc_len, self.max_dec_len)

        if do_infer:
            total_eval_batch_size = mpu.get_data_parallel_world_size() * args.batch_size
            total_data_num = math.ceil(len(self.data) / total_eval_batch_size) * total_eval_batch_size
            while len(self.data) < total_data_num:
                tmp = self.data[0].copy()
                tmp["idx"] = -1
                self.data.append(tmp)

        print_str = "Path: {} | Ratio:{} | Max enc len: {} | Max dec len: {} | Data num: {}".format(path, ratio, self.max_enc_len, self.max_dec_len, len(self.data))
        print_rank_0(print_str)
        save_rank_0(args, print_str)

    def process_data(self):
        raise NotImplementedError

    def add_prompt_ids(self, data, max_enc_len, max_dec_len):
        enc_prompt_ids = [-i-1 for i in range(self.prompt_len)]
        dec_prompt_ids = [-i-1 for i in range(self.prompt_config["dec"]["prompt_len"])]
        pad_ids = [self.tokenizer.pad_id for _ in range(self.prompt_config["dec"]["prompt_len"])]

        for d in data:
            # front
            d["enc_input_ids"] = enc_prompt_ids + d["enc_input_ids"]
            # front + mid
            #d["enc_input_ids"] = enc_prompt_ids[:self.prompt_len//2] + d["enc_input_ids"][:d["premise_end_id"]] + \
            #enc_prompt_ids[self.prompt_len//2:] + d["enc_input_ids"][d["premise_end_id"]:]
            d["dec_input_ids"] = dec_prompt_ids + d["dec_input_ids"]
            d["label_ids"] = pad_ids + d["label_ids"]

        max_enc_len += self.prompt_len
        max_dec_len += self.prompt_config["dec"]["prompt_len"]

        return data, max_enc_len, max_dec_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate(self, samples):
        bs = len(samples)
        model_data = {
            "enc_input_ids": torch.ones(bs, self.max_enc_len, dtype=torch.long) * self.pad_id,
            "enc_attention_mask": torch.zeros(bs, 1, self.max_enc_len, self.max_enc_len),
            "dec_attention_mask": torch.zeros(bs, 1, self.max_dec_len, self.max_dec_len),
            "cross_attention_mask": torch.zeros(bs, 1, self.max_dec_len, self.max_enc_len),
            "dec_input_ids": torch.ones(bs, self.max_dec_len, dtype=torch.long) * self.pad_id
        }
        if not self.do_infer:
            no_model_data = {
                "idx": torch.zeros(bs, dtype=torch.long),
                "labels": torch.ones(bs, self.max_dec_len, dtype=torch.long) * self.pad_id,
                "loss_mask": torch.zeros(bs, self.max_dec_len),
            }
        else:
            no_model_data = {
                "idx": torch.zeros(bs, dtype=torch.long),
            }

        for i, samp in enumerate(samples):
            enc_len, dec_len = len(samp["enc_input_ids"]), len(samp["dec_input_ids"])
            model_data["enc_input_ids"][i][:enc_len] = torch.tensor(samp["enc_input_ids"], dtype=torch.long)
            model_data["dec_input_ids"][i][:dec_len] = torch.tensor(samp["dec_input_ids"], dtype=torch.long)
            model_data["enc_attention_mask"][i][0, :enc_len, :enc_len] = 1.0
            model_data["dec_attention_mask"][i][0, :dec_len, :dec_len] = torch.tril(torch.ones(dec_len, dec_len))
            model_data["cross_attention_mask"][i][0, :dec_len, :enc_len] = 1.0
            no_model_data["idx"][i] = samp["idx"]
            # no_model_data["lang"][i] = samp["lang"]
            if not self.do_infer:
                no_model_data["labels"][i][:len(samp["label_ids"])] = torch.tensor(samp["label_ids"], dtype=torch.long)
                if self.prompt_config is not None:
                    no_model_data["loss_mask"][i][self.prompt_config["dec"]["prompt_len"]:len(samp["label_ids"])] = 1.0
                else:
                    no_model_data["loss_mask"][i][:len(samp["label_ids"])] = 1.0
        
        if self.args.fp16:
            model_data["enc_attention_mask"] = model_data["enc_attention_mask"].half()
            model_data["dec_attention_mask"] = model_data["dec_attention_mask"].half()
            model_data["cross_attention_mask"] = model_data["cross_attention_mask"].half()

        # print(model_data)
        # print(no_model_data)
        return model_data, no_model_data


class XNLIDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None, lang=None):
        super(XNLIDataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer, prompt_config, lang)

    def process_data(self):

        self.label_word_map = {
            "entailment": "tail",
            "contradiction": "contra",
            "neutral": "neutral", 
            'contradictory': "contra"         
        }

        labels = self.tokenizer.encode("tail contra neutral")
        data = []
        enc_sizes = []
        dec_sizes = []
        
        with open(self.path, "r") as f:
            lines = f.readlines()
            if self.split == 'train': lines = lines[1:]

        for line in lines[:int(self.ratio * len(lines))]:
            if self.split == 'train':
                premise, hypo, label = line[:-1].split('\t')
                label = self.label_word_map[label]
                lang = "train_lang"
            else:
                d = json.loads(line)
                lang, premise, hypo, label = d["language"], d["sentence1"], d["sentence2"], d["gold_label"]
                label = self.label_word_map[label]
                if lang != self.lang : continue
                
            if self.do_infer or label:
                # premise_prefix = self.tokenizer.encode("xnli: premise: ")
                # hypo_prefix = self.tokenizer.encode(" hypothesis: ")
                # context = premise_prefix + self.tokenizer.encode(premise)[:self.enc_seq_length // 2 - len(premise_prefix)] + hypo_prefix + self.tokenizer.encode(hypo)[:self.enc_seq_length // 2 - len(hypo_prefix)]
                premise = self.tokenizer.encode(premise)[:self.enc_seq_length // 2]
                hypo = self.tokenizer.encode(hypo)[:self.enc_seq_length // 2]
                context = premise + hypo + [1]
                target = [1, self.tokenizer.get_sentinel_id(0)] + (self.tokenizer.encode(label) if not self.do_infer else [self.tokenizer.pad_id])
                #target = [1, self.tokenizer.get_sentinel_id(0)] + (self.tokenizer.encode(label)[1:2] if not self.do_infer else [self.tokenizer.pad_id])
                if self.add_target_post:
                    target += [self.tokenizer.get_sentinel_id(1)]
                data.append({
                    "idx": self.idx,
                    "enc_input_ids": context,
                    "dec_input_ids": target[:-1],
                    "label_ids": target[1:],
                    "premise_end_id": len(premise)
                })
                enc_sizes.append(len(context))
                dec_sizes.append(len(target)-1)
                self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len

class PAWSXDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None, lang=None):
        super(PAWSXDataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer, prompt_config, lang)

    def process_data(self):

        data = []
        enc_sizes = []
        dec_sizes = []

        if self.split != 'train':
            self.path = self.path.replace('/en/', '/{}/'.format(self.lang))

        with open(self.path, "r") as f:
            lines = f.readlines()
            lines = lines[1:]

        #train_langs = ['en', 'de', 'es', 'fr', 'ja', 'ko', 'zh']
        train_langs = ['en']
        if self.split == 'train':
            for lg in train_langs[1:]:
                tmp = open(self.path.replace('en/train.tsv', "{}/translated_train.tsv".format(lg))).readlines()
                lines.extend(tmp)
                
        for line in lines[:int(self.ratio * len(lines))]:
            _, sent1, sent2, label = line[:-1].split('\t')
                
            if self.do_infer or label:
                context = self.tokenizer.encode(' '.join(['sentence1:', sent1, 'sentence2:', sent2]))
                context = context[:self.enc_seq_length-1] + [1]
                target = [0, self.tokenizer.get_sentinel_id(0)] + (self.tokenizer.encode(label) if not self.do_infer else [self.tokenizer.pad_id])

                data.append({
                    "idx": self.idx,
                    "enc_input_ids": context,
                    "dec_input_ids": target[:-1],
                    "label_ids": target[1:],
                    "premise_end_id": 0
                })
                enc_sizes.append(len(context))
                dec_sizes.append(len(target)-1)
                self.idx += 1

        # multi-node
        #random.shuffle(data)

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len