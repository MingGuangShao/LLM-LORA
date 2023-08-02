import torch
  
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BloomForCausalLM, GenerationConfig

BASE_MODEL = "bigscience/bloomz-7b1-mt"
LORA_WEIGHTS = "../bloom_lora"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16, # 加载半精度
        device_map={"":1}, # 指定GPU 0
    )
model.eval()

# 加载LoRA权重
model = PeftModel.from_pretrained(model, LORA_WEIGHTS, torch_dtype=torch.float16)
# model = PeftModel.from_pretrained(model, LORA_WEIGHTS)
model.half()


while True:
    inputs = input("请输入标题:")
    #instruction = "In this task, you need extract the most attractive short promotional vocabulary term from the product title. use Traditional Chinese."
    instruction = "抽取標題中的營銷詞\n\ninput:Waroom|現貨 熱賣 🇰🇷正韓復古格紋寬鬆內裡西裝外套|女裝|雙口袋|長袖小西服|西裝外套|大衣|格子外套 920\noutput:熱賣、現貨"
    instruction = instruction + '\n\n' +  "input:{}\noutput:".format(inputs)
    encodings = tokenizer(instruction, max_length=1024, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids=encodings["input_ids"], max_new_tokens=1024)
    preds = tokenizer.decode(outputs[0])
    print(preds.split("output:")[2])
