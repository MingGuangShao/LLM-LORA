import os
import sys
import json

import argparse
import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BloomForCausalLM, GenerationConfig

prompt_fomat = '### Instruction:\n{}' + '\n\n' +  '### Input:{}\n\n### Response:\n'


def inference(
    base_model: str = "",
    lora_weights: str = "",
    # the infer data, if not exists, infer the default instructions in code
    data_path: str = "",
    load_8bit: bool = False,
):
    
    # 加载预测数据
    inference_data = []
    if data_path != "":
        with open(data_path + '/data.json', "r") as f:
            inference_data = json.load(f)
            
    else:
        inference_data = [
            {"instruction":"抽取標題中的營銷詞", "input":"短版小西裝外套女秋裝2022年新款薄款洋氣時尚高級西裝正裝上衣潮"},
            {"instruction":"抽取標題中的營銷詞", "input":"【高品質 】西裝外套 小外套 西裝女 西裝 外套女 春秋新款 韓版 時尚緊身短版休閒女士西裝上衣 新品"},
            {"instruction":"抽取標題中的營銷詞", "input":"韓版 蝴蝶結雪紡衫女 寬鬆短袖上衣 超仙法式洋氣素色上衣 女生衣著 夏季新款"},
            {"instruction":"抽取標題中的營銷詞", "input":"多功能護理墊 輕薄超吸收💕papa母嬰💕看護墊 寶寶防尿墊 尿墊 拋棄式產墊 尿布墊 月經墊 寵物尿墊 產褥墊"},
        ]

    
    
    # 加载基模型
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16, # 加载半精度
        device_map="auto", # 指定GPU 0
    )

    # 加载LoRA权重
    model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.float16)
    if not load_8bit:
        model.half()  # seems to fix bugs for some users.
    # eval 模式
    model.eval()

    # 打开use_cache加速
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        prompt,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=256,
        **kwargs,
    ):
        encodings = tokenizer(prompt, max_length=1024, return_tensors="pt")
        input_ids = encodings["input_ids"].to("cuda")
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                # generation_config=generation_config,
                # return_dict_in_generate=True,
                # output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        pred = tokenizer.decode(generation_output[0])
        return pred

        
    for d in inference_data:
        instruction = d["instruction"]
        input = d["input"]
        model_output = evaluate(prompt_fomat.format(instruction, input))
        print("###model output###")
        print(model_output)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Name of the dataset')
    parser.add_argument('--lora_model', type=str, help='Name of the lora_model')
    args = parser.parse_args()
    
    
    filepath = os.path.split(os.path.realpath(__file__))[0]
    parentpath = os.path.abspath(os.path.join(filepath,".."))
    
    data_path = ""
    if args.dataset:
        data_path = "{}/datasets/{}".format(parentpath, args.dataset)
        
    inference(
        base_model = "bigscience/bloomz-7b1-mt", 
        lora_weights = "{}/scripts/{}".format(parentpath, args.lora_model),
        data_path = data_path, 
    )
