from os import POSIX_FADV_SEQUENTIAL, replace
from preprocess_chid_finetune import process_one_sent
import torch
import json
import re
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
import gzip
import csv

class T5Dataset(Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, ratio=1, prefix=None, add_target_post=False, cache_path=None, prompt_config=None, lang=None):
        self.args = args
        self.tokenizer = tokenizer
        self.ratio = ratio
        self.path = path
        self.pad_id = tokenizer.pad_id
        self.prefix = prefix
        self.prefix_ids = self.tokenizer.encode(prefix) if prefix is not None else []
        self.enc_seq_length = args.enc_seq_length - len(self.prefix_ids)
        self.add_target_post=add_target_post
        self.idx = 0
        self.label = 0
        self.prompt_config = prompt_config
        if self.prompt_config:
            self.prompt_len = self.prompt_config["enc"]["prompt_len"]
            self.align_prompt_len = self.prompt_config["enc"]["align_prompt_len"]
        self.lang = lang

        if cache_path is not None:
            cache_path = os.path.join(cache_path, "cache_{}_{}.pkl".format(path.replace("/", "_"), ratio))
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    self.data, self.max_enc_len = pickle.load(f)
            else:
                self.data, self.max_enc_len = self.process_data()
                with open(cache_path, "wb") as f:
                    pickle.dump((self.data, self.max_enc_len), f)
        else:
            self.data, self.max_enc_len = self.process_data()

        if prompt_config is not None:
            self.data, self.max_enc_len = self.add_prompt_ids(self.data, self.max_enc_len)

        print_str = "Path: {} | Ratio:{} | Max enc len: {} | Data num: {}".format(path, ratio, self.max_enc_len, len(self.data))
        print_rank_0(print_str)
        save_rank_0(args, print_str)

    def process_data(self):
        raise NotImplementedError

    def add_prompt_ids(self, data, max_enc_len):
        enc_prompt_ids = [-i-1 for i in range(self.prompt_len)]
        align_prompt_ids = [-i-self.prompt_len-1 for i in range(self.align_prompt_len)]

        for d in data:
            # front
            #d["enc_input_ids"] = enc_prompt_ids + d["enc_input_ids"][:d["premise_end_id"]] + d["enc_input_ids"][d["premise_end_id"]:]
            # front + mid
            #d["enc_input_ids"] = d["enc_input_ids"] 
            d["enc_input_ids"] = enc_prompt_ids[:self.prompt_len//2] + d["enc_input_ids"] + enc_prompt_ids[self.prompt_len//2:] + align_prompt_ids
            d["labels"] = d["labels"]

        max_enc_len += self.prompt_len + self.align_prompt_len

        return data, max_enc_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate(self, samples):
        bs = len(samples)
        model_data = {
            "enc_input_ids": torch.ones(bs, self.max_enc_len, dtype=torch.long) * self.pad_id,
            "enc_attention_mask": torch.zeros(bs, 1, self.max_enc_len, self.max_enc_len),
        }
        no_model_data = {
            "idx": torch.zeros(bs, dtype=torch.long),
            "labels": torch.ones(bs, 1, dtype=torch.long) * self.pad_id,
        }

        for i, samp in enumerate(samples):
            #print(self.tokenizer.decode([i for i in samp["enc_input_ids"] if i>0]))
            enc_len = len(samp["enc_input_ids"])
            model_data["enc_input_ids"][i][:enc_len] = torch.tensor(samp["enc_input_ids"], dtype=torch.long)
            model_data["enc_attention_mask"][i][0, :enc_len, :enc_len] = 1.0
            no_model_data["idx"][i] = samp["idx"]
            no_model_data["labels"][i] = samp["labels"]

        
        if self.args.fp16:
            model_data["enc_attention_mask"] = model_data["enc_attention_mask"].half()

        # print(model_data)
        # print(no_model_data)
        return model_data, no_model_data


class DictDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, ratio=1, prefix=None, add_target_post=False, cache_path=None, prompt_config=None):
        super(DictDataset, self).__init__(args, tokenizer, path, ratio, prefix, add_target_post, cache_path, prompt_config)

    def process_data(self):

        data = []
        enc_sizes = []
        
        with open(self.path, "r") as f:
            lines = f.readlines()
            random.shuffle(lines)

        for line in lines[:int(self.ratio * len(lines))]:
            words = line[:-1].split('\t')
            if len(words) == 1: words = words[0].split()
            for word in words:
                context = self.tokenizer.encode(word)
                data.append({
                    "idx": self.idx,
                    "enc_input_ids": context,
                    "labels": self.label,
                })
                enc_sizes.append(len(context))
                self.idx += 1

            self.label += 1

        max_enc_len = max(enc_sizes)

        return data, max_enc_len
    
class AllNLIDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, ratio=1, prefix=None, add_target_post=False, cache_path=None, prompt_config=None):
        super(AllNLIDataset, self).__init__(args, tokenizer, path, ratio, prefix, add_target_post, cache_path, prompt_config)

    def process_data(self):

        data = []
        enc_sizes = []

        label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
        train_samples = []
        with gzip.open(self.path, 'rt', encoding='utf8') as fIn:
            reader = list(csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE))
            random.shuffle(reader)
            for row in reader:
                if row['split'] == 'train':
                    label_id = label2int[row['label']]
                for sent in [row['sentence1'], row['sentence2']]:
                    context = self.tokenizer.encode(sent)[:self.enc_seq_length]
                    data.append({
                        "idx": self.idx,
                        "enc_input_ids": context,
                        "labels": label_id,
                    })
                    enc_sizes.append(len(context))
                    self.idx += 1

        max_enc_len = max(enc_sizes)
        return data, max_enc_len


# class DictDataset(T5Dataset):
#     def __init__(self, args, tokenizer: EncDecTokenizer, path, ratio=1, prefix=None, add_target_post=False, cache_path=None, prompt_config=None, lang=None):
#         super(DictDataset, self).__init__(args, tokenizer, path, ratio, prefix, add_target_post, cache_path, prompt_config, lang)

#     def get_processed_data(file_path):
#         lines = []
#         with open(file_path) as file:
#             for line in file:
#             text = re.findall("<seg id=(.*?)>(.*?)</seg>", line)
#             if text: lines.append(text[0][1])
#         return lines

#     def process_data(self):
#         target_languages = ['cs', 'de', 'ja', 'pl', 'ru', 'ta', 'zh']
#         for language in target_languages:
#             lang1 = "sgm/newstest2020-{}en-ref.en.sgm".format(language)
#             lang2 = "sgm/newstest2020-{}en-src.{}.sgm".format(language, language)
#             test_evaluator = TranslationEvaluator(self.get_processed_data(lang1), self.get_processed_data(lang2), batch_size=inference_batch_size, name=language, show_progress_bar=False, layer=layer)
#             evaluators.append(test_evaluator)


class TatoebaDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, ratio=1, prefix=None, add_target_post=False, cache_path=None, prompt_config=None, lang=None):
        super(TatoebaDataset, self).__init__(args, tokenizer, path, ratio, prefix, add_target_post, cache_path, prompt_config, lang)

    def process_data(self):
        data = []
        enc_sizes = []
        
        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines[:int(self.ratio * len(lines))]:
            line = line.strip()
            # print(line)
            context = self.tokenizer.encode(line) + [1]
            # import transformers
            # tokenizer = transformers.T5Tokenizer.from_pretrained('google/mt5-large')
            # inputs = tokenizer.encode(line)
            # print("current", inputs)
            # print("huggingface", context)
            data.append({
                "idx": self.idx,
                "enc_input_ids": context,
                "labels": self.label,
            })
            enc_sizes.append(len(context))
            self.idx += 1
            self.label += 1

        max_enc_len = max(enc_sizes)

        return data, max_enc_len


