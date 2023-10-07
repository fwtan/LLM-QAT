from tqdm import tqdm
import torch, os, argparse, gc
from datasets import load_dataset
from transformers import PreTrainedModel, AutoTokenizer, AutoConfig
from models.configuration_llama import LlamaConfig
from models.modeling_llama_quant import (
    LlamaForCausalLM as LlamaForCausalLMQuant,
)
from models.modeling_llama_quant import LlamaRMSNorm
from models.utils_quant import  QuantizeLinear, QuantizerWrapper, QuantizeBMM, SymQuantizer, AsymQuantizer


def print_model_size(model, include_buffers=False):
    # https://discuss.pytorch.org/t/finding-model-size/130275
    param_size = 0
    param_cnt = 0
    for param in model.parameters():
        param_cnt += param.nelement()
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    buffer_cnt = 0
    if include_buffers:
        for buffer in model.buffers():
            buffer_cnt += buffer.nelement()
            buffer_size += buffer.nelement() * buffer.element_size()
    cnt_all = (param_cnt + buffer_cnt) / 10**9
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Model size: {:.3f}B/{:.3f}MB'.format(cnt_all, size_all_mb))


class Evaluator:
    def __init__(self, dataset, tokenizer, max_length=2048):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        # # tokenize the dataset
        # def tokenize_function(examples):
        #     example = self.tokenizer(examples['text'])
        #     return example
        # self.dataset = self.dataset.map(tokenize_function, batched=True)
        # self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        device = next(model.buffers()).device
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        latency = 0

        for batch in tqdm(self.dataset):
            # input_ids = batch['input_ids'].to(device).unsqueeze(0)
            input_ids = torch.tensor(self.tokenizer(batch['text']).input_ids)[:self.max_length]
            input_ids = input_ids.to(device).unsqueeze(0)
            label = input_ids[:, -1]
            pad_len = self.max_length - input_ids.shape[1]
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=0)
            torch.cuda.synchronize()
            start.record()
            if isinstance(model, PreTrainedModel):
                outputs = model(input_ids)
            else:
                outputs = model(input_ids[0])
            end.record()
            torch.cuda.synchronize()
            latency += start.elapsed_time(end)
            if isinstance(model, PreTrainedModel):
                logits = outputs.logits
            else:
                logits = outputs
                logits = logits.unsqueeze(0)
            last_token_logits = logits[:, -2-pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()

        acc = hit / total
        latency = latency / len(self.dataset)
        return acc, latency


parser = argparse.ArgumentParser()
parser.add_argument('--hf_path', type=str, default='checkpoints/llama/2B_smoothed', help='path of the hf model')
parser.add_argument('--stat_path', type=str, default='checkpoints/llama/2B_smoothed/act_scales_for_quant.pth', help='path of the act stat')
args = parser.parse_args()


seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


def main():
    tokenizer = AutoTokenizer.from_pretrained(args.hf_path, use_fast=False, legacy=False)
    dataset = load_dataset('lambada', split='validation[:1000]')
    evaluator = Evaluator(dataset, tokenizer, 256)

    # HF model
    hf_device = 'cuda:0'
    hf_config = LlamaConfig.from_pretrained(args.hf_path)

    hf_config.w_bits  = 8
    hf_config.a_bits  = 8
    hf_config.kv_bits = 8
    
    model_hf = LlamaForCausalLMQuant.from_pretrained(pretrained_model_name_or_path=args.hf_path, config=hf_config, torch_dtype=torch.float32, low_cpu_mem_usage=True, device_map=hf_device)
    print_model_size(model_hf, True)
    act_stat = torch.load(args.stat_path)

    for name, m in model_hf.named_modules():
        if isinstance(m, (QuantizeLinear, QuantizeBMM, LlamaRMSNorm)):
            m.set_act_scale(act_stat[name])
            if 'down_proj' in name:
                if m.input_quantizer is not None:
                    m.input_quantizer.bitwidth = 16
                if m.weight_quantizer is not None:
                    m.weight_quantizer.bitwidth = 16
                if m.output_quantizer is not None:
                    m.output_quantizer.bitwidth = 16
            if 'layernorm' in name:
                if m.input_quantizer is not None:
                    m.input_quantizer.bitwidth = 16
                if m.weight_quantizer is not None:
                    m.weight_quantizer.bitwidth = 16
            if 'o_proj' in name:
                if m.output_quantizer is not None:
                    m.output_quantizer.bitwidth = 16
        

    with torch.no_grad():
        acc_hf, latency_hf = evaluator.evaluate(model_hf)
    print(f'HF accuracy: {acc_hf:.3f}, per-sample latency: {latency_hf:.3f}ms')
    del model_hf
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()