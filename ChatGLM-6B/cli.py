from transformers import AutoTokenizer, AutoModel
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'model','ChatYuan-large-v2')
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
history = []
print("starting")
while True:
    query = input("\n用户：")
    if query == "stop":
        break
    if query == "clear":
        history = []
        os.system('clear')
        continue
    response, history = model.chat(tokenizer, query, history=history)
    print(f"小元：{response}")


if __name__ == "__main__":
    main()
