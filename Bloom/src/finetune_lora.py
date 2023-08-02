import os
import sys
import argparse
from typing import List, Union

import torch
import wandb #日志记录
import transformers
from datasets import load_dataset, load_from_disk
from loguru import logger

import torch.nn as nn
import bitsandbytes as bnb

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
# from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402
from transformers import AutoTokenizer, AutoModelForCausalLM
from processing import process_dataset


def generate_prompt(
    instruction: str, 
    input: Union[None, str] = None, 
    label: Union[None, str] = None,
) -> str:
    
    prompt_input =  '### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n'
    prompt_no_input =  '### Instruction:\n{instruction}\n\n### Response:\n'
        
    if input:
        res = prompt_input.format(instruction=instruction, input=input)
    else:
        res = prompt_no_input.format(instruction=instruction)
    
    if label:
        res = f"{res}{label}"
    
    return res


def train(
    # model/data params
    base_model: str = "",
    data_path: str = "",
    output_dir: str = "",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 3,
    num_epochs: int = 1,
    learning_rate: float = 3e-5,
    cutoff_len: int = 512,
    val_set_size: int = 500,
    eval_steps: int = 100,
    save_steps: int = 100,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "query_key_value"
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    logging_steps=2,
    bits=16, # 参数
    
):
    # 打印参数
    logger.info(
        f"Training Bloomz-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"eval_steps: {eval_steps}\n"
        f"save_steps: {save_steps}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"logging_steps: {logging_steps}\n"
        f"bits: {bits}\n"
    )

    # wandb打印loss
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_run_name) > 0:
        os.environ["WANDB_RUN_NAME"] = wandb_run_name
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

        
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=True,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"]
        )
        tokenized_full_prompt = tokenize(full_prompt)
        
        # loss 训练时，是否将prompt计算在内
        if not train_on_inputs:
            user_prompt = generate_prompt(
                data_point["instruction"],
                data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            # 将 instruction 和 input 部分mask掉，达到loss只考虑输出的目的
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]
        return tokenized_full_prompt


    # 加载数据
    if os.path.exists(data_path + '/dataset'):
        data = load_from_disk(data_path + '/dataset')
    else:
        # data = load_dataset("json", data_files=data_path)
        process_dataset(data_path = data_path + '/data.json', save_path = data_path + '/dataset')
    
    # 这里注意一下
    gradient_accumulation_steps = batch_size // micro_batch_size

    # world_size 总共有多少个进程参与训练，如果你有两台服务器，每台服务器有四张卡，那么 World Size 就是 2 x 4 = 8。
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model, 
        torch_dtype="auto",
        device_map=device_map
    )

    
    if bits < 16:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
        model = prepare_model_for_int8_training(model) #对load_in_8_bit适配用于提升lora的稳定性
    
    
    # 从另一篇文章中看到的
    # 模型显存占用分成两个部分，一部分是静态显存基本由模型参数量级决定
    # 另一部分是动态显存在向前传播的过程中每个样本的每个神经元都会计算激活值并存储，用于向后传播时的梯度计算，这部分和batchsize以及参数量级相关
    # 8bit量化优化的是静态显存，而梯度检查优化的是动态显存
    
    # from_pretrained中的load_in_8bit参数是bitsandbytes库赋予的能力，会把加载模型转化成混合8bit的量化模型，注意这里的8bit模型量化只用于模型推理，
    # 通过量化optimizer state降低训练时显存的时8bit优化器是另一个功能不要搞混
    
    #模型量化本质是对浮点参数进行压缩的同时，降低压缩带来的误差。 8-bit quantization是把原始FP32（4字节）压缩到Int8（1字节）也就是1/4的显存占用。
    # 如上加载后会发现除lora层外的多数层被转化成int类型如下
    # model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map=device_map)

    # 当然压缩方式肯定不是直接四舍五入，那样会带来巨大的精度压缩损失。常见的量化方案有absolute-maximum和zero-point，它们的差异只是rescale的方式不同
    # prepare_model_for_int8_training是对在Lora微调中使用LLM.int8()进行了适配用来提高训练的稳定性，主要包括
    # layer norm层保留FP32精度  和 输出层保留FP32精度保证解码时随机sample的差异性
    
    # prepare_model_for_int8_training函数还做了一件事就是设置gradient_checkpointing=True，这是另一个时间换空间的技巧。
    # gradient checkpoint的实现是在向前传播的过程中使用torch.no_grad()不去存储中间激活值，降低动态显存的占用。
    # 而只是保存输入和激活函数，当进行反向传播的时候，会重新获取输入和激活函数计算激活值用于梯度计算。因此向前传播会计算两遍，所以需要更多的训练时间。
    
    # use_cache设置为False，是因为和gradient checkpoint存在冲突。因为use_cache是对解码速度的优化，在解码器解码时，存储每一步输出的hidden-state用于下一步的输入，
    # 而因为开启了gradient checkpoint，中间激活值不会存储，因此use_cahe=False。
    
    # 这个文章写的特别好 https://cloud.tencent.com/developer/article/2276508
    # 这个文章还介绍了不同量化带来的速度差异  https://blog.csdn.net/zhouzhou0929/article/details/131140225
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # tokenizer.pad_token_id = (
    #     0  # unk. we want this to be different from the eos token
    # )
    # tokenizer.padding_side = "left"  # Allow batched inference


    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules, # target_modules中的作用目标名在不同模型中的名字是不一样的
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM", # 这是LoraConfig的父类PeftConfig中的参数，设定任务的类型
        # inference_mode=False,
    )
    
    model = get_peft_model(model, config)
    


    # 从 ckpt加载lora权重去继续训练，对PeftModel中的Lora权重进行加载
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            logger.info(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            logger.info(f"Checkpoint {checkpoint_name} not found")

    # 打印可训练的参数
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.


    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt, remove_columns=train_val['train'].column_names)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt, remove_columns=train_val['test'].column_names)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt, remove_columns=train_val['train'].column_names)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    # v100上可能不支持混合精度，导致报错 RuntimeError: expected scalar type Half but found Float
    # 因此关闭fp16=True, 这会导致显存增加
    
    '''
    This code creates a TrainingArguments object which specifies various settings and hyperparameters for training the model. These include:

    gradient_accumulation_steps: Number of updates steps to accumulate gradients before performing a backward/update pass.
    warmup_steps: Number of warmup steps for the optimizer.
    max_steps: The total number of training steps to perform.
    learning_rate: The learning rate for the optimizer.
    fp16: Use 16-bit precision for training.    
    '''
    
    '''
    DataCollatorForSeq2Seq is a class from the Transformers library that creates batches of input/output sequences for sequence-to-sequence (seq2seq) models. 
    In this code, a DataCollatorForSeq2Seq object is instantiated with the following parameters:

    pad_to_multiple_of: An integer representing the maximum sequence length, rounded up to the nearest multiple of this value.
    padding: A boolean indicating whether to pad the sequences to the specified maximum length.
    '''
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # warmup_steps=1000,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=False, # 是否使用fp16的16为混合精度训练 而不是使用fp32的32为训练, 实际上置为true会占用更多的显存 https://github.com/huggingface/peft/issues/381
            logging_steps=logging_steps,#和 wandb有关
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_steps if val_set_size > 0 else None,
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    # 这几步要在trainer实例化之后进行
    '''
    After instantiating the Trainer, the code sets use_cache to False in the model's config, and creates a state_dict for the model using the get_peft_model_state_dict() function, 
    which prepares the model for training using low-precision arithmetic.

    Then, the torch.compile() function is called on the model, which compiles the model's computation graph and prepares it for training using PyTorch 2.
    '''
    # Silence the warnings. Please re-enable for inference!
    model.config.use_cache = False # 因为没使用8bit量化，可以打开；如果使用8bit量化，则需置为False；但这里同样先置为false

    # 函数为模型创建一个state_dict，该函数为使用低精度算法进行训练的模型做准备
    # 在实例化训练器之后，代码在模型的配置中将use_cache设置为False，并使用get_peft_model_state_dict()函数为模型创建一个state_dict，该函数为使用低精度算法进行训练的模型做准备
    if bits < 16:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict
            )
        ).__get__(model, type(model))
    
    # 不使用，避免隐藏的问题
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)
    

    # with torch.autocast("cuda"):
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    model.save_pretrained(output_dir)

    logger.info(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--deepspeed', type=str)
    parser.add_argument('--dataset', type=str, help='Name of the dataset')
    parser.add_argument('--lora_model', type=str, help='Name of the lora_model')
    parser.add_argument('--wandb_project', type=str, help='Name of the wandb_project')
    parser.add_argument('--wandb_run_name', type=str, help='Name of the wandb_run')
    args = parser.parse_args()
    
    
    filepath = os.path.split(os.path.realpath(__file__))[0]
    parentpath = os.path.abspath(os.path.join(filepath,".."))
        
    train(
        base_model = "bigscience/bloomz-7b1-mt", 
        data_path = "{}/datasets/{}".format(parentpath, args.dataset), 
        output_dir = "{}/scripts/{}".format(parentpath, args.lora_model), 
        batch_size = 128,
        micro_batch_size = 3,
        num_epochs = 1,
        learning_rate = 1e-4,
        cutoff_len = 512,
        val_set_size = 500,
        eval_steps = 100,
        save_steps = 100,
        lora_r = 8,
        lora_alpha = 16,
        lora_dropout = 0.05,
        wandb_project = args.wandb_project,
        wandb_run_name = args.wandb_run_name,
        logging_steps = 2,
        train_on_inputs=False,
    )