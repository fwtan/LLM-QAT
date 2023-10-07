import torch, os, gc
import torch.nn as nn
from copy import deepcopy
from datasets import load_dataset
from collections import defaultdict
from functools import partial
import numpy as np
import os.path as osp
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from models.configuration_llama import LlamaConfig
from models.modeling_llama_quant import (
    LlamaForCausalLM as LlamaForCausalLMQuant,
)
from models.modeling_llama_quant import LlamaRMSNorm
from models.utils_quant import  QuantizeLinear, QuantizerWrapper, QuantizeBMM, SymQuantizer, AsymQuantizer


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--hf_path',    type=str, default='checkpoints/llama/2B_smoothed', help='path of the hf model')
parser.add_argument("--output_dir", type=str, default='results/stat')
parser.add_argument('--calib_data', type=str, default='pileval', help='the calibration data')
parser.add_argument('--calib_path', type=str, default='data/pile/val.jsonl.zst', help='the calibration data')
parser.add_argument('--num_calib_samples', type=int, default=1024, help='num of calibration samples')
parser.add_argument('--calib_seq_len', type=int, default=2048, help='max seq len for the calibration samples')
args = parser.parse_args()


seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


@torch.no_grad()
def get_act_scales_for_quantization(model, tokenizer, dataset, num_samples, seq_len, pad_to_max_length=False):
    model.eval()
    device = next(model.parameters()).device
    act_dict = defaultdict(dict)

    def stat_io_hook(m, xx, yy, name):
        # input
        x = xx[0] if isinstance(xx, tuple) else xx
        x = x.to(torch.float32).detach()
        min_x, max_x = x.min().item(), x.max().item()
        if name not in act_dict or "input" not in act_dict[name]:
            act_dict[name]["input"] = [min_x, max_x]
        else:
            act_dict[name]["input"] = [min(act_dict[name]["input"][0], min_x), max(act_dict[name]["input"][1], max_x)]
        # output
        y = yy[0] if isinstance(yy, tuple) else yy
        y = y.to(torch.float32).detach()
        min_y, max_y = y.min().item(), y.max().item()
        if name not in act_dict or "output" not in act_dict[name]:
            act_dict[name]["output"] = [min_y, max_y]
        else:
            act_dict[name]["output"] = [min(act_dict[name]["output"][0], min_y), max(act_dict[name]["output"][1], max_y)]
        # weight
        if isinstance(m, (QuantizeBMM)):
            w = xx[1].to(torch.float32).detach()
            min_w, max_w = w.min().item(), w.max().item()
            if name not in act_dict or "weight" not in act_dict[name]:
                act_dict[name]["input2"] = [min_w, max_w]
            else:
                act_dict[name]["input2"] = [min(act_dict[name]["weight"][0], min_w), max(act_dict[name]["weight"][1], max_w)]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (QuantizeLinear, LlamaRMSNorm, QuantizeBMM)):
            hooks.append(m.register_forward_hook(partial(stat_io_hook, name=name)))

    samples = []
    for i in tqdm(range(num_samples)):
        if "text" in dataset[i]:        line = dataset[i]["text"]
        elif "content" in dataset[i]:   line = dataset[i]["content"]
        elif "ctx" in dataset[i]:       line = dataset[i]["ctx"]
        else:                           raise NotImplementedError
        input_ids = tokenizer(line.strip(), return_tensors="pt", max_length=args.calib_seq_len, truncation=True).input_ids
        if pad_to_max_length:
            input_ids = torch.nn.functional.pad(input_ids, (0, seq_len - input_ids.shape[1]), value=tokenizer.pad_token_id)
        samples.append(input_ids)
        # following the stable diffusion demo from qualcomm, add some samples with random indices
        samples.append(torch.randint(tokenizer.bos_token_id+1, tokenizer.vocab_size-1, size=(1, args.calib_seq_len), dtype=torch.int32))

    print("Collecting activation scales for quantization...")
    pbar = tqdm(range(len(samples)))
    for i in pbar:
        model(samples[i].to(device))
        mean_scale = np.mean([v["input"] for v in act_dict.values()])
        pbar.set_description(f"Mean input scale: {mean_scale:.2f}")
    for hook in hooks:
        hook.remove()

    return act_dict


def main():
    # HF model
    hf_device = 'cuda:0'
    hf_config = LlamaConfig.from_pretrained(args.hf_path)
    model_hf = LlamaForCausalLMQuant.from_pretrained(pretrained_model_name_or_path=args.hf_path, config=hf_config, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=hf_device)

    # calib data
    if args.calib_data == 'pileval':
        dataset = load_dataset("json", data_files=args.calib_path, split="train")
    elif args.calib_data == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split="train")
    else:
        dataset = load_dataset(args.calib_data, split="validation")
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_path, use_fast=False, legacy=False)
    os.makedirs(args.output_dir, exist_ok=True)

    for x in model_hf.parameters(): 
        x.requires_grad = False
    
    dataset = dataset.shuffle(seed=seed+1)
    act_scales_for_quant = get_act_scales_for_quantization(model_hf, tokenizer, dataset, args.num_calib_samples, args.calib_seq_len, False)
    torch.save(act_scales_for_quant, osp.join(args.output_dir, 'act_scales_for_quant.pth'))


if __name__ == '__main__':
    main()