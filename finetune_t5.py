# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain Enc-Dec"""

# Flag to use Pytorch ddp which uses overlapping communication and computation.
USE_TORCH_DDP = False

from datetime import datetime
from infer_funcs import infer_xnli
import os
import random
import math
import numpy as np
import torch
import json
from tqdm import tqdm
import shutil

import deepspeed

from arguments import get_args
from data_utils.tokenization_enc_dec import EncDecTokenizer
from fp16 import FP16_Module
from fp16 import FP16_Optimizer
from learning_rates import AnnealingLR
from model import EncDecModel, EncDecConfig
from model import enc_dec_get_params_for_weight_decay_optimization, enc_dec_get_params_for_prompt_optimization

if USE_TORCH_DDP:
    from torch.nn.parallel.distributed import DistributedDataParallel as DDP
else:
    from model import DistributedDataParallel as DDP
import mpu
from apex.optimizers import FusedAdam as Adam
from utils import Timers
from utils import save_prompt
from utils import load_checkpoint
from utils import report_memory
from utils import print_args
from utils import print_rank_0, save_rank_0
import torch.distributed as dist

from data.samplers import DistributedBatchSampler, RandomSampler

from T5Dataset import XNLIDataset, T5Dataset, PAWSXDataset

import torch.nn.functional as F

import time


def get_model(args, vocab_size, prompt_config=None):
    """Build the model."""

    print_rank_0('building Enc-Dec model ...')
    config = EncDecConfig.from_json_file(args.model_config)
    config.vocab_size = vocab_size
    config.tuning_type = args.tuning_type
    model = EncDecModel(config,
                        parallel_output=True,
                        checkpoint_activations=args.checkpoint_activations,
                        checkpoint_num_layers=args.checkpoint_num_layers,
                        prompt_config=prompt_config)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if args.deepspeed and args.fp16:
        model.half()

    # GPU allocation.
    model.cuda(torch.cuda.current_device())
    # if args.prompt_tune:
    #     model.init_prompt_embeds()

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    if USE_TORCH_DDP:
        i = torch.cuda.current_device()
        model = DDP(model, device_ids=[i], output_device=i,
                    process_group=mpu.get_data_parallel_group())
    else:
        model = DDP(model)

    # torch.cuda.synchronize()
    # model.module.module.print_prompt_embeds()

    return model


def get_optimizer(model, args, prompt_config=None):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (DDP, FP16_Module)):
        model = model.module
    if args.prompt_tune:
        param_groups = enc_dec_get_params_for_prompt_optimization(args, model, args.wd)
    else:
        param_groups = enc_dec_get_params_for_weight_decay_optimization(model)
    
    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False

    if args.cpu_optimizer:
        if args.cpu_torch_adam:
            cpu_adam_optimizer = torch.optim.Adam
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(param_groups,
                        lr=args.lr, weight_decay=args.weight_decay)
    else:
        # Use FusedAdam.
        optimizer = Adam(param_groups,
                         lr=args.lr, weight_decay=args.weight_decay)
        # Use Adafactor
        # optimizer = Adafactor(param_groups, lr=args.lr, weight_decay=args.weight_decay, 
        # scale_parameter=False, relative_step=False)

    print(f'Optimizer = {optimizer.__class__.__name__}')
    if args.deepspeed:
        # fp16 wrapper is not required for DeepSpeed.
        return optimizer

    # Wrap into fp16 optimizer.
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale': args.min_scale,
                                       'delayed_shift': args.hysteresis})

    if torch.distributed.get_rank() == 0:
        print(optimizer.param_groups)

    return optimizer


def get_learning_rate_scheduler(optimizer, args):
    """Build the learning rate scheduler."""

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    num_iters = max(1, num_iters)
    init_step = -1
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step,
                               gradient_accumulation_steps=args.gradient_accumulation_steps)

    return lr_scheduler


def setup_model_and_optimizer(args, vocab_size, ds_config, prompt_config=None):
    """Setup model and optimizer."""

    model = get_model(args, vocab_size, prompt_config)
    optimizer = get_optimizer(model, args, prompt_config)
    lr_scheduler = get_learning_rate_scheduler(optimizer, args)

    if args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            lr_scheduler=lr_scheduler,
            mpu=mpu,
            dist_init_required=False,
            config_params=ds_config
        )

    print(args.load)
    if args.load is not None:
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args)
    else:
        args.iteration = 0

    if args.prompt_tune:
        model.module.module.module.init_prompt_embeds()
        if args.prompt_path: 
            device = model.module.module.module.encoder.prompt_embeds.weight.device
            prompt_weight = torch.load(args.prompt_path)['encoder.prompt_embeds.weight'].to(device)
            model.module.module.module.encoder.prompt_embeds.weight = torch.nn.Parameter(prompt_weight)

    return model, optimizer, lr_scheduler


def forward_step(args, model_batch, no_model_batch, model, device, keep_enc_hidden=False, do_infer=False):
    for k in model_batch:
        model_batch[k] = model_batch[k].to(device)
    for k in no_model_batch:
        no_model_batch[k] = no_model_batch[k].to(device)

    if keep_enc_hidden:
        enc_outputs = model(**model_batch, only_encoder=True)
        enc_hidden_states = enc_outputs["encoder_last_hidden_state"]
        output = model(**model_batch, enc_hidden_states=enc_hidden_states)
    else:
        output = model(**model_batch)
    
    logits = output["lm_logits"]
    forw_out = {
        "logits": logits
    }
    if keep_enc_hidden:
        forw_out["enc_hidden_states"] = enc_hidden_states
    
    # tnews
    # label_tokens = [555, 2637, 1116, 2477, 317, 1464, 1129, 821, 678, 241, 1123, 2858, 1608, 323, 261]

    # if mpu.get_model_parallel_rank() == 0:
    #     logits_bak = logits[:, -1, label_tokens].clone()
    #     logits[:, -1, :] = float("-inf")
    #     logits[:, -1, label_tokens] = logits_bak
    # else:
    #     logits[:, -1, :] = float("-inf")

    if not do_infer:
        losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), no_model_batch["labels"])
        loss_mask = no_model_batch["loss_mask"]
        losses = (losses * loss_mask).sum(-1) / loss_mask.sum(-1)
        loss = losses.mean()

        forw_out["loss"] = loss
    
    return forw_out


def backward_step(args, loss, model, optimizer):
    # backward
    if args.deepspeed:
        model.backward(loss)
    else:
        optimizer.zero_grad()
        if args.fp16:
            optimizer.backward(loss, update_master_grads=False)
        else:
            loss.backward()

    # Update master gradients.
    if not args.deepspeed:
        if args.fp16:
            optimizer.update_master_grads()

        # Clipping gradients helps prevent the exploding gradient.
        if args.clip_grad > 0:
            if not args.fp16:
                mpu.clip_grad_norm(model.parameters(), args.clip_grad)
            else:
                optimizer.clip_master_grads(args.clip_grad)


def train(args, data_config, tokenizer, model, optimizer, lr_scheduler,
          train_dataset, train_dataloader, prompt_config, device):
    """Train the model."""

    eval_func = data_config[args.data_name]["eval_func"]

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_loss = 0.0

    step, global_step = 1, 1

    best_accs = []
    best_acc = 0
    early_stopping = 0

    for e in range(args.epochs):
        model.train()
        for model_batch, no_model_batch in train_dataloader:

            forw_out = forward_step(args, model_batch, no_model_batch, model, device)
            loss = forw_out["loss"]
            
            if torch.distributed.get_rank() == 0:
                print(loss)

            backward_step(args, loss, model, optimizer)

            # Update losses.
            total_loss += loss.item()

            if args.deepspeed:
                model.step()
            else:
                optimizer.step()
                if not (args.fp16 and optimizer.overflow):
                    lr_scheduler.step()

            # Logging.
            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                learning_rate = optimizer.param_groups[0]['lr']
                avg_lm_loss = total_loss / (args.log_interval * args.gradient_accumulation_steps)
                log_string = 'epoch {:3d}/{:3d} |'.format(e, args.epochs)
                log_string += ' global iteration {:8d}/{:8d} |'.format(global_step, args.train_iters)
                log_string += ' learning rate {:.3} |'.format(learning_rate)
                log_string += ' lm loss {:.6} |'.format(avg_lm_loss)
                if args.fp16:
                    log_string += ' loss scale {:.1f} |'.format(optimizer.cur_scale if args.deepspeed else optimizer.loss_scale)
                print_rank_0(log_string)
                save_rank_0(args, log_string)
                total_loss = 0.0

            # Checkpointing
            # if args.save and args.save_interval and global_step % args.save_interval == 0 and step % args.gradient_accumulation_steps == 0:
            #     save_checkpoint(global_step, model, optimizer, lr_scheduler, args)

            # Evaluation
            if global_step > 0 and args.eval_interval and global_step % args.eval_interval == 0 and step % args.gradient_accumulation_steps == 0 and args.do_valid:
                prefix = 'iteration {} | '.format(global_step)
                avg_eval_loss, avg_eval_acc = 0, 0
                langs = args.eval_lang.split(',')
                for l in langs:
                    eval_dataloader, eval_dataset = load_data(args, data_config, 'dev', tokenizer, prompt_config, ratio=1, lang=l)
                    eval_func = data_config[args.data_name]["eval_func"]
                    loss, acc = eval_func(args, tokenizer, data_config, eval_dataset, eval_dataloader, model, device, mode="dev")
                    avg_eval_loss += loss
                    avg_eval_acc += acc
                    log_string = "Dev result: prefix: {} lang: {} loss: {:.6} | acc: {:.4}".format(prefix, l, loss, acc)
                    print_rank_0(log_string)
                    save_rank_0(args, log_string)

                avg_eval_loss, avg_eval_acc = avg_eval_loss / len(langs), avg_eval_acc / len(langs)
                log_string = "Dev result: prefix: {} lang: {} loss: {:.6} | acc: {:.4}".format(prefix, 'avg', avg_eval_loss, avg_eval_acc)
                print_rank_0(log_string)
                save_rank_0(args, log_string)

                # if best_acc > avg_eval_acc:
                #     early_stopping += 1
                #     if early_stopping >= 10: exit()
                # else:
                #     best_acc = avg_eval_acc
                #     early_stopping = 0

                if args.max_save > 0:
                    i = 0
                    while i < len(best_accs):
                        if best_accs[i][1] < avg_eval_acc:
                            break
                        i += 1
                    if len(best_accs) < args.max_save or i < len(best_accs):
                        best_accs.insert(i, (global_step, avg_eval_acc))
                        if len(best_accs) > args.max_save:
                            step_to_be_rm, acc_to_be_rm = best_accs[-1]
                            if torch.distributed.get_rank() == 0:
                                shutil.rmtree(os.path.join(args.save, "acc_{}_{:.3}".format(step_to_be_rm, acc_to_be_rm)))
                        save_prompt(global_step, model, optimizer, lr_scheduler, args, save_dir=os.path.join(args.save, "acc_{}_{:.3}".format(global_step, avg_eval_acc)))
                        best_accs = best_accs[:args.max_save]

            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1

    return global_step


def evaluate(args, tokenizer: EncDecTokenizer, data_config, eval_dataset, eval_data_loader, model, device, mode='dev'):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0.0
    step = 0

    all_idx = []
    all_preds = []
    all_labels = []
    all_langs = []

    with torch.no_grad():
        for model_batch, no_model_batch in eval_data_loader:
            forw_out = forward_step(args, model_batch, no_model_batch, model, device, do_infer=(mode=="infer"))
            loss = forw_out["loss"].item() if "loss" in forw_out else 0
            total_loss += loss

            logits_list = [torch.zeros_like(forw_out["logits"]) for _ in range(mpu.get_model_parallel_world_size())]
            torch.distributed.all_gather(logits_list, forw_out["logits"], mpu.get_model_parallel_group())

            gathered_logits = torch.cat(logits_list, dim=-1)

            pred_token_logits = gathered_logits[:, 1, :]
            preds = torch.argmax(pred_token_logits, dim=-1)
            gathered_preds = [torch.zeros_like(preds) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(gathered_preds, preds.contiguous(), mpu.get_data_parallel_group())
            all_preds.extend(gathered_preds)
            
            gathered_idx = [torch.zeros_like(no_model_batch["idx"]) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(gathered_idx, no_model_batch["idx"].contiguous(), mpu.get_data_parallel_group())
            all_idx.extend(gathered_idx)

            
            if mode != "infer":
                labels = no_model_batch["labels"][:, 1]
                gathered_labels = [torch.zeros_like(labels) for _ in range(mpu.get_data_parallel_world_size())]
                torch.distributed.all_gather(gathered_labels, labels.contiguous(), mpu.get_data_parallel_group())
                all_labels.extend(gathered_labels)

            step += 1

    total_loss /= step

    all_idx = torch.cat(all_idx, dim=0).cpu().tolist()
    all_preds = torch.cat(all_preds, dim=0).cpu().tolist()

    model.train()

    if mode == "infer":
        return data_config[args.data_name]["infer_func"](args, tokenizer, all_idx, all_preds)
    else:
        all_labels = torch.cat(all_labels, dim=0).cpu().tolist()

        acc = sum([int(p == l) for p, l in zip(all_preds, all_labels)]) / len(all_preds)

        if torch.distributed.get_rank() == 0:
            with open(args.log_file+str(acc), 'w') as fout:
                for p, l in zip(all_preds, all_labels):
                    fout.write("{} {}\n".format(p, l))

        return total_loss, acc


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        
    if top_p > 0.0:
        #convert to 1D
        logits=logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        #going back to 2D
        logits=logits.view(1, -1).contiguous()

    return logits


def evaluate_gen(args, tokenizer: EncDecTokenizer, data_config, eval_dataset: T5Dataset, eval_data_loader, model, device, mode="dev"):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0.0
    step = 0

    all_preds = []
    all_labels = []
    all_idx = []

    with torch.no_grad():
        for model_batch, no_model_batch in eval_data_loader:
            
            forw_out = forward_step(args, model_batch, no_model_batch, model, device, keep_enc_hidden=True, do_infer=(mode=="infer"))
            loss = forw_out["loss"].item() if "loss" in forw_out else 0
            total_loss += loss

            enc_hidden_states = forw_out["enc_hidden_states"]

            # for generating responses
            # we only use the <go> token, so truncate other tokens
            dec_input_ids = model_batch['dec_input_ids'][..., :1]
            dec_attention_mask = model_batch['dec_attention_mask'][..., :1, :1]
            # # we use past_key_values, so only the current token mask is needed
            cross_attention_mask = model_batch['cross_attention_mask'][..., :1, :]

            unfinished_sents = model_batch['enc_input_ids'].new(model_batch['enc_input_ids'].size(0)).fill_(1)
            output_ids = model_batch['enc_input_ids'].new_zeros([model_batch['enc_input_ids'].size(0), 0])
            past_key_values = None

            gen_len = 0
            while gen_len < args.dec_seq_length:
                if unfinished_sents.max() == 0:
                    tokens_to_add = tokenizer.pad_id * (1 - unfinished_sents)
                    output_ids = torch.cat([output_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

                else:
                    dec_outputs = model(
                        dec_input_ids=dec_input_ids,
                        dec_attention_mask=dec_attention_mask,
                        cross_attention_mask=cross_attention_mask,
                        enc_hidden_states=enc_hidden_states,
                        past_key_values=past_key_values,
                    )
                    lm_logits = dec_outputs["lm_logits"]
                    past_key_values = dec_outputs['past_key_values']

                    gathered_lm_logits = [torch.zeros_like(lm_logits).to(device) for _ in range(mpu.get_model_parallel_world_size())]
                    torch.distributed.all_gather(gathered_lm_logits, lm_logits.data, mpu.get_model_parallel_group())

                    lm_logits = torch.cat(gathered_lm_logits, dim=-1)

                    next_token_logits = lm_logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1)
                    # next_token_logscores = top_k_logits(next_token_logits, top_k=args.top_k, top_p=args.top_p)
                    # probs = F.softmax(next_token_logscores, dim=-1)
                    # next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                    tokens_to_add = next_token * unfinished_sents + tokenizer.pad_id * (1 - unfinished_sents)

                    dec_input_ids = tokens_to_add.unsqueeze(-1)
                    output_ids = torch.cat([output_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                    # let the current token attend to all previous tokens
                    dec_attention_mask = torch.cat([dec_attention_mask, dec_attention_mask[:, :, :, -1:]], dim=-1)

                gen_len += 1
                unfinished_sents.mul_(tokens_to_add.ne(tokenizer.get_sentinel_id(1)).long())
            
            gathered_preds = [torch.zeros_like(output_ids) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(gathered_preds, output_ids, mpu.get_data_parallel_group())
            all_preds.extend(gathered_preds)
            
            gathered_idx = [torch.zeros_like(no_model_batch["idx"]) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(gathered_idx, no_model_batch["idx"].contiguous(), mpu.get_data_parallel_group())
            all_idx.extend(gathered_idx)

            if mode != "infer":
                gathered_labels = [torch.zeros_like(no_model_batch["labels"]) for _ in range(mpu.get_data_parallel_world_size())]
                torch.distributed.all_gather(gathered_labels, no_model_batch["labels"], mpu.get_data_parallel_group())
                all_labels.extend(gathered_labels)

            step += 1

    total_loss /= step

    all_idx = torch.cat(all_idx, dim=0).cpu().tolist()
    all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
    all_preds = [e[:e.index(tokenizer.pad_id)] if tokenizer.pad_id in e else e for e in all_preds]
    if args.data_name in ["wsc2"]:
        all_preds = [tokenizer.decode(p[1:-1]) for p in all_preds]
        all_preds = [int(p in d["cand_ids"] or d["cand_ids"] in p) for p, d in zip(all_preds, eval_dataset.data)]

    if mode == "infer":
        return data_config[args.data_name]["infer_func"](args, tokenizer, all_idx, all_preds)
    else:
        if args.data_name == "wsc2":
            all_labels = [d["truth"] for d in eval_dataset.data]
        else:
            all_labels = torch.cat(all_labels, dim=0).cpu().tolist()
            all_labels = [e[:e.index(tokenizer.pad_id)] if tokenizer.pad_id in e else e for e in all_labels]
        
        if args.data_name in ["cmrc"]:
            acc = cmrc_metric(tokenizer, all_preds, eval_dataset.data)
        else:
            acc = sum([int(p == l) for p, l in zip(all_preds, all_labels)]) / len(all_preds)

        return total_loss, acc


def wsc2_metric(tokenizer: EncDecTokenizer, all_preds, all_labels, all_truth):
    all_preds = [p[1:-1] for p in all_preds]
    all_labels = [l[1:-1] for l in all_labels]
    res = []
    for p, l, t in zip(all_preds, all_labels, all_truth):
        p = tokenizer.decode(p)
        l = tokenizer.decode(l)
        pp = int((p in l) or (l in p))
        res.append(int(pp == t))

    acc = sum(res) / len(res)
    return acc


def cmrc_metric(tokenizer: EncDecTokenizer, all_preds, data):
    print("Doing cmrc metric")        
    all_preds = [tokenizer.decode(p[1:-1]) for p in all_preds]
    res = [int(p in d["truth"]) for p, d in zip(all_preds, data)]

    acc = sum(res) / len(res)
    return acc


def evaluate_and_print_results(tokenizer, prefix, data_iterator, model,
                               args, timers, verbose=False):
    """Helper function to evaluate and dump results on screen."""
    lm_loss = evaluate(tokenizer, data_iterator, model, args, timers, verbose)
    lm_ppl = math.exp(min(20, lm_loss))
    string = '-' * 100 + "\n"
    string += ' validation loss at {} | '.format(prefix)
    string += 'LM loss: {:.6} | '.format(lm_loss)
    string += 'LM PPL: {:.6}'.format(lm_ppl)
    length = len(string) + 1
    string = '-' * length + "\n" + string + "\n" + '-' * length
    print_rank_0(string)
    save_rank_0(args, string)

    return lm_loss


def set_deepspeed_activation_checkpointing(args):

    deepspeed.checkpointing.configure(mpu, deepspeed_config=args.deepspeed_config, num_checkpoints=args.num_layers)
    mpu.checkpoint = deepspeed.checkpointing.checkpoint
    mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
    mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    deepspeed.init_distributed()

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

    # Optional DeepSpeed Activation Checkpointing Features
    #
    if args.deepspeed and args.deepspeed_activation_checkpointing:
        set_deepspeed_activation_checkpointing(args)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)


def load_data(args, data_config, data_type, tokenizer, prompt_config=None, ratio=1, drop_last=True, do_infer=False, lang=None):

    dataset = data_config[args.data_name]["dataset"](
        args,
        tokenizer,
        data_config[args.data_name][data_type+'_file'],
        data_type,
        ratio=ratio,
        cache_path=data_config[args.data_name]["cache_path"],
        do_infer=do_infer,
        prompt_config=prompt_config,
        lang=lang)

    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = args.batch_size * world_size
    num_workers = args.num_workers

    if data_type == 'train':
        sampler = RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
        global_batch_size = args.eval_batch_size
    batch_sampler = DistributedBatchSampler(sampler=sampler,
                                            batch_size=global_batch_size,
                                            drop_last=drop_last,
                                            rank=rank,
                                            world_size=world_size)

    data_loader = torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True,
                                       collate_fn=dataset.collate)

    # Torch dataloader.
    return data_loader, dataset


def main():
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()

    os.makedirs(args.save, exist_ok=True)

    # Pytorch distributed.
    initialize_distributed(args)
    if torch.distributed.get_rank() == 0:
        print('Pretrain Enc-Dec model')
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)

    # Random seeds for reproducability.
    set_random_seed(args.seed)
    device = torch.cuda.current_device()

    # setup tokenizer
    tokenizer = EncDecTokenizer(args.tokenizer_path)
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size

    prompt_config = None
    if args.prompt_tune:
        with open(args.prompt_config, "r") as f:
            prompt_config = json.load(f)
            for t in ["enc", "dec"]:
                prompt_config[t]["init_ids"] = tokenizer.encode(prompt_config[t]["init_tokens"])
                pad_num = prompt_config[t]["prompt_len"] - len(prompt_config[t]["init_ids"])
                # prompt_config[t]["init_ids"].extend(tokenizer.convert_tokens_to_ids([prompt_config[t]["default_init_token"] for _ in range(pad_num)]))
                prompt_config[t]["init_ids"].extend([100*(i+1) for i in range(pad_num)])
                prompt_config[t]["init_ids"] = torch.tensor(prompt_config[t]["init_ids"], dtype=torch.long).to(device)

    # Model, optimizer, and learning rate.
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, tokenizer.vocab_size, ds_config, prompt_config)
    data_config = {
        "xnli": {
            "dataset": XNLIDataset,
            "eval_func": evaluate,
            "cache_path": None,
            "infer_func": infer_xnli,
            "train_file": os.path.join(args.data_path, "multinli/multinli.train.{}.tsv".format(args.train_lang)),
            "dev_file": os.path.join(args.data_path, "xnli/xnli.dev.en.jsonl"),
            "test_file": os.path.join(args.data_path, "xnli/xnli.test.en.jsonl")
        },
        "pawsx": {
            "dataset": PAWSXDataset,
            "eval_func": evaluate,
            "cache_path": None,
            "infer_func": infer_xnli,
            "train_file": os.path.join(args.data_path, "x-final/en/train.tsv"),
            "dev_file": os.path.join(args.data_path, "x-final/en/dev_2k.tsv"),
            "test_file": os.path.join(args.data_path, "x-final/en/test_2k.tsv")
        }
    }

    if args.do_train:
        train_dataloader, train_dataset = load_data(args, data_config, 'train', tokenizer, prompt_config, ratio=1.0)
        train(args, data_config, tokenizer, model, optimizer, lr_scheduler, train_dataset, train_dataloader, prompt_config, device)

    if args.do_eval:
        avg_eval_loss, avg_eval_acc = 0, 0
        langs = args.eval_lang.split(',')
        for l in langs:
            eval_dataloader, eval_dataset = load_data(args, data_config, 'test', tokenizer, prompt_config, ratio=1, lang=l)
            eval_func = data_config[args.data_name]["eval_func"]
            loss, acc = eval_func(args, tokenizer, data_config, eval_dataset, eval_dataloader, model, device, mode="test")
            avg_eval_loss += loss
            avg_eval_acc += acc
            log_string = "Test result: lang: {} loss: {:.6} | acc: {:.4}".format(l, loss, acc)
            print_rank_0(log_string)
            save_rank_0(args, log_string)

        avg_eval_loss, avg_eval_acc = avg_eval_loss / len(langs), avg_eval_acc / len(langs)
        log_string = "Test result: lang: {} loss: {:.6} | acc: {:.4}".format('avg', avg_eval_loss, avg_eval_acc)
        print_rank_0(log_string)
        save_rank_0(args, log_string)

    if args.do_infer:
        infer_dataloader, infer_dataset = load_data(args, data_config, "test", tokenizer, prompt_config, ratio=1, drop_last=True, do_infer=True)
        eval_func = data_config[args.data_name]["eval_func"]

        sample_num = eval_func(args, tokenizer, data_config, infer_dataset, infer_dataloader, model, device, mode="infer")

        log_string = "Inferenced {} samples".format(sample_num)
        print_rank_0(log_string)
        save_rank_0(args, log_string)


if __name__ == "__main__":
    main()
