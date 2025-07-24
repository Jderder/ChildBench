import os
import json
import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as T

# 配置参数
MODEL_PATH = "xxx"
IMAGE_DIR = "xxx"
JSON_PATH = "xxx"
OUTPUT_PATH = "xxx"

def build_transform(input_size=448):
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

def load_images(image_files, input_size=448):
    transform = build_transform(input_size)
    pixel_values = []
    for img_file in image_files:
        img_path = os.path.join(IMAGE_DIR, img_file)
        image = Image.open(img_path).convert('RGB')
        pixel_values.append(transform(image))
    pixel_values = torch.stack(pixel_values)  # [N, 3, H, W]
    return pixel_values

def make_prompt(question, answer_type, options=None):
    # 构造 options_str
    options_str = ""
    if options:
        options_str = " Options: " + " ".join(options)
    # 按题型构造 prompt
    if answer_type == 'choice' and options_str:
        prompt = f'You are currently an expert in a wide range of early childhood education topics as well as fundamental visual reasoning themes.' \
            f"Given the Image, a Question, your task is to answer the correct answer without any explanation. Note that you only need to choose one option from all options without explaining any reason. Input: Question: {question}{options_str}. \nOutput:"
    else:
        prompt = f'You are currently an expert in a wide range of early childhood education topics as well as fundamental visual reasoning themes.' \
            f"Given the Image, a Question, your task is to answer the correct answer without any explanation. Input: Question: {question}. \nOutput:"
    return prompt

def main():
    # 加载模型
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)

    # 加载数据
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    total_count = 0
    right_count = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for idx, item in enumerate(data, 1):
            question_id = item["question_id"]
            question = item["question"]
            answer_type = item.get("answer_type", "")
            options = item.get("options", None)
            input_images = item["input_image"]
            answer = item.get("answer", "")
            category = item.get("category", "")

            # 实时进度打印
            print(f"正在处理第{idx}/{len(data)}个，ID: {question_id}")

            # 加载多图
            pixel_values = load_images(input_images).to(torch.bfloat16).cuda()

            # 构造 prompt
            prompt = make_prompt(question, answer_type, options)

            # 推理
            generation_config = dict(max_new_tokens=512, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            output = model.chat(tokenizer, pixel_values, prompt, generation_config)
            output = output.strip()
            print(f"模型原始回答: {output}")
            answer_upper = answer.strip().upper()

            # 判分
            result = 0
            if answer_type == 'choice':
                # 取第一个字母
                output_upper = output[0].upper() if output else ''
                if output_upper in answer_upper or answer_upper in output_upper:
                    result = 1
            else:
                output_upper = output.upper()
                if output_upper == answer_upper:
                    result = 1
            if result == 1:
                right_count += 1
            total_count += 1

            # 写入结果
            result_json = {
                'id': question_id,
                'image': input_images[0],
                'result': result,
                'output': output_upper,
                'answer': answer,
                'category': category
            }
            fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')

            # 实时准确率打印
            print(f"当前累计准确率: {right_count}/{total_count} = {right_count/total_count if total_count else 0:.4f}")

    print(f"总数: {total_count}，正确数: {right_count}，准确率: {right_count/total_count if total_count else 0:.4f}")

if __name__ == "__main__":
    main() 