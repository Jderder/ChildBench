import os
import json
import torch
from PIL import Image
from modelscope import AutoConfig, AutoModel, AutoTokenizer
import re

# 路径配置
MODEL_PATH = 'xxx'
JSON_PATH = 'xxx'
OUTPUT_PATH = 'xxx'
IMAGE_ROOT = 'xxx'  # 这里改成你的图片实际存放目录

# 加载模型
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_PATH,
    attn_implementation='sdpa',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
_ = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
processor = model.init_processor(tokenizer)

def replace_image_placeholders(question, num_images):
    """
    将<image>、<image_1>、<image_2>等依次替换为<|image|>，返回替换后的文本和图片顺序
    """
    # 匹配所有<image>、<image_1>、<image_2>...
    pattern = re.compile(r'<image(?:_\d*)?>')
    matches = list(pattern.finditer(question))
    if len(matches) != num_images:
        print(f"警告：图片数与占位符数不一致！")
    # 替换为<|image|>
    new_question = pattern.sub('<|image|>', question)
    return new_question

def make_prompt(question, answer_type, options=None):
    options_str = ""
    if options:
        options_str = " Options: " + " ".join(options)
    if answer_type == 'choice' and options_str:
        prompt = f'You are currently an expert in a wide range of early childhood education topics as well as fundamental visual reasoning themes.Given the Image, a Question, your task is to answer the correct answer without any explanation. Note that you only need to choose one option from all options without explaining any reason. Input: Question: {question}{options_str}. \nOutput:'
    else:
        prompt = f'You are currently an expert in a wide range of early childhood education topics as well as fundamental visual reasoning themes.Given the Image, a Question, your task is to answer the correct answer without any explanation. Input: Question: {question}. \nOutput:'
    return prompt

def main():
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

            print(f"正在处理第{idx}/{len(data)}个，ID: {question_id}")

            # 加载所有图片
            images = [Image.open(os.path.join(IMAGE_ROOT, img_name)).convert('RGB') for img_name in input_images]

            # 替换问题中的<image>、<image_1>等为<|image|>
            prompt = replace_image_placeholders(question, len(images))
            # 拼接完整prompt
            if answer_type == 'choice' and options:
                options_str = " Options: " + " ".join(options)
                prompt = f"{prompt}{options_str}. \nOutput:"
            else:
                prompt = f"{prompt}. \nOutput:"

            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ""}
            ]
            inputs = processor(messages, images=images, videos=None)
            inputs.to('cuda')
            inputs.update({
                'tokenizer': tokenizer,
                'max_new_tokens': 100,
                'decode_text': True,
            })

            # 推理
            g = model.generate(**inputs)
            if isinstance(g, list):
                output = g[0].strip()
            else:
                output = g.strip()
            print(f"模型原始回答: {output}")

            # 判分
            result = 0
            answer_upper = answer.strip().upper()
            if answer_type == 'choice':
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

            print(f"当前累计准确率: {right_count}/{total_count} = {right_count/total_count if total_count else 0:.4f}")

    print(f"总数: {total_count}，正确数: {right_count}，准确率: {right_count/total_count if total_count else 0:.4f}")

if __name__ == "__main__":
    main()