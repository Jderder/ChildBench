import os
import json
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 路径配置
MODEL_PATH = 'xxx'
JSON_PATH = 'xxx'
OUTPUT_PATH = 'xxx'
IMAGE_ROOT = 'xxx'  # 改成你的图片实际目录

# 加载模型和分词器
model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    attn_implementation='sdpa',
    torch_dtype=torch.float16
)
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# TARGET_SIZE = (448, 448)  # 或 (224, 224)，根据模型实际需求调整



def make_prompt(question, answer_type, options=None):
    options_str = ""
    if options:
        options_str = " Options: " + " ".join(options)
    if answer_type == 'choice' and options_str:
        prompt = f"You are currently an expert in a wide range of early childhood education topics as well as fundamental visual reasoning themes." \
                 f"Given the Image, a Question, your task is to answer the correct answer without any explanation. Note that you only need to choose one option from all options without explaining any reason.Input: Question: {question}{options_str}."
    else:
        prompt = f"You are currently an expert in a wide range of early childhood education topics as well as fundamental visual reasoning themes." \
                 f"Given the Image, a Question, your task is to answer the correct answer without any explanation.Input: Question: {question}."
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

            # 支持多图输入
            images = [
                Image.open(os.path.join(IMAGE_ROOT, img_name)).convert('RGB')
                for img_name in input_images
            ]

            prompt = make_prompt(question, answer_type, options)
            # content: [image1, image2, ..., prompt]
            msgs = [{'role': 'user', 'content': images + [prompt]}]

            # 推理
            try:
                res = model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=tokenizer
                )
                output = res if isinstance(res, str) else str(res)
            except Exception as e:
                print(f"推理出错: {e}，图片列表: {input_images}")
                output = ""

            print(f"模型原始回答: {output}")

            # 判分
            result = 0
            answer_upper = answer.strip().upper()
            if answer_type == 'choice' and options:
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
                'image': input_images,  # 多图时为图片名列表
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