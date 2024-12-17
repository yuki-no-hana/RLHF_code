#########################################################################################################
# 模型初始化
#########################################################################################################
import os
import sys
import fire
import torch
# from peft import PeftModel
import copy
import json
import time

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

os.environ['WANDB_MODE'] = 'dryrun'

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(torch.cuda.device_count())

base_model = "models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590" or os.environ.get("BASE_MODEL", "")
prompt_template = "llama"

print("We are using device", device)
print("We are using model", base_model)
config_kwargs = {
    "output_hidden_states": True
}

config = AutoConfig.from_pretrained(base_model, **config_kwargs)

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,  
        device_map="auto",
        cache_dir='./',
        config=config
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        base_model, low_cpu_mem_usage=True
    )


###################################################################################################
# 数据处理
###################################################################################################
from datasets import load_dataset
from trl.data_utils import maybe_apply_chat_template
from trl.trainer.utils import DPODataCollatorWithPadding, truncate_right

def tokenize_row(feature, is_encoder_decoder, tokenizer):
    """Tokenize a single row from a DPO specific dataset."""
    if not is_encoder_decoder:
        batch = tokenizer(feature["prompt"], add_special_tokens=False)
        # Add BOS token to head of prompt. Avoid adding if it's already there
        if tokenizer.bos_token_id is not None:
            prompt_len_input_ids = len(batch["input_ids"])
            if prompt_len_input_ids == 0 or tokenizer.bos_token_id != batch["input_ids"][0]:
                batch["input_ids"] = [tokenizer.bos_token_id] + batch["input_ids"]
                batch["attention_mask"] = [1] + batch["attention_mask"]
    else:
        batch = tokenizer(feature["prompt"], add_special_tokens=True)
    batch = {f"prompt_{key}": value for key, value in batch.items()}
    return batch
    

# 导入数据集
train_dataset = load_dataset('json', data_files='trl_data.json')
train_dataset = train_dataset['train'] # 该数据集没有answer

# 制作data_collator
data_collator = DPODataCollatorWithPadding(pad_token_id=tokenizer.pad_token_id)
print("the padding token for model is", tokenizer.pad_token_id)

# 这里假设batch size = 2
batch_size = 4
batch_data = train_dataset[:batch_size]

# 数据处理部分
prompts = batch_data["prompt"]
inputs = [{k: v[i] for k, v in batch_data.items()} for i in range(batch_size)]
inputs = [maybe_apply_chat_template(x, tokenizer) for x in inputs] # 添加收尾结束字符
# tokenize
inputs = [tokenize_row(x, model.config.is_encoder_decoder, tokenizer) for x in inputs] # 这一步不会pad
# 制作一个batch
inputs = data_collator(inputs)
print("the pre-processed input is ", inputs)
inputs_in_word = tokenizer.decode(inputs['prompt_input_ids'][1], skip_special_tokens=True)
print(inputs_in_word)



########################################################################################################
# 找到重要的token
########################################################################################################

from transformers import GenerationConfig
from trl.trainer.utils import truncate_right


def get_important_token(model, inputs, tokenizer, generation_config=None):
    prompt_ids = inputs["prompt_input_ids"].to(device)
    prompt_mask = inputs["prompt_attention_mask"].to(device)
    input_length = torch.sum(prompt_mask, dim=1)
    num_examples, context_length = inputs["prompt_input_ids"].shape

    if generation_config is None:
        generation_config = GenerationConfig(
            max_new_tokens=64,
            temperature=0.9,
            top_k=0,
            top_p=1.0,
            do_sample=True,
        )
    output = model.generate(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        generation_config=generation_config
    )

    generated_text = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(batch_size)]
    #for i in generated_text:
    #    print(i)

    # 截断回答，之后的部分用pad_token_id补齐
    completion_ids = output[:, context_length:]
    completion_ids, completion_mask = truncate_right(
        completion_ids, tokenizer.eos_token_id, tokenizer.pad_token_id
    ) # eos_token_id and pad_token_id is 2
    prompt_completion_ids = torch.cat((prompt_ids, completion_ids), dim=1)
    prompt_completion_mask = torch.cat((prompt_mask, completion_mask), dim=1)
    print("the size of prompt_ids, completion_ids, prompt_mask, completion_mask is", prompt_ids.shape, completion_ids.shape, prompt_mask.shape, completion_mask.shape)

    # 得到token的倒数第二层embedding
    output = model(prompt_completion_ids, attention_mask=prompt_completion_mask)
    print("the shape of input and its mask for getting embedding is", prompt_completion_ids.shape, prompt_completion_mask.shape)
    embedding = output.hidden_states[-2] # shape is 2, 168, 4096
    input_embedding = embedding[:, :context_length]
    output_embedding = embedding[:, context_length - 1: -1]

    # input_embedding = input_embedding * prompt_mask.unsqueeze(-1)
    avg_output_embedding = output_embedding * completion_mask.unsqueeze(-1) # size 2 104 4096
    avg_output_embedding = torch.mean(avg_output_embedding, dim=1).unsqueeze(-1) # size 2 4096 1
    print("the averaged output embedding size is", avg_output_embedding.shape)

    #计算attention
    att = torch.matmul(input_embedding, avg_output_embedding).squeeze(-1)
    value, indices = torch.topk(att, int(att.shape[-1] / 3), dim=1)
    #print(indices)
    # attention转换成attention mask
    attention_mask = torch.zeros(prompt_ids.shape, dtype=torch.int64).to(device)
    attention_mask.scatter_(1, indices, torch.ones_like(indices))
    #print(attention_mask)

    # 展示attention的部分是否合理
    attention_input_ids = prompt_ids * attention_mask
    attention_input_ids = torch.where(attention_input_ids == 0, torch.tensor(259), attention_input_ids)

    completion_text = [tokenizer.decode(attention_input_ids[i], skip_special_tokens=True) for i in range(batch_size)]
    for i in range(len(completion_text)):
        print(generated_text[i])
        input()
        print(attention_input_ids[i])
        print(completion_text[i])
        input()



get_important_token(model, inputs, tokenizer)
#prompt = "Hello!"
#input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
#outputs = model(prompt_ids, max_length=600, num_return_sequences=1)
#print(outputs.past_key_values)
#generated_text = tokenizer.decode(outputs.past_key_values[0], skip_special_tokens=True)
#print(input_ids)
#print(generated_text)
