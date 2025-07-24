import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_id = "xxx"


IMAGE_DIR = "xxx"
FILE_PATH = "xxx"
RESULT_FILE_PATH = "xxx"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

count = 0
right_count = 0

def make_prompt(question, answer_type, options=None):
    options_str = ""
    if options:
        options_str = " Options: " + " ".join(options)
    if answer_type == 'choice' and options_str:
        prompt = f"You are currently an expert in a wide range of early childhood education topics as well as fundamental visual reasoning themes."\
        f"Given the Image, a Question, your task is to answer the correct answer without any explanation. Note that you only need to choose one option from all options without explaining any reason.Input: Question: {question}{options_str}."
    else:
        prompt = f"You are currently an expert in a wide range of early childhood education topics as well as fundamental visual reasoning themes."\
                f"Given the Image, a Question, your task is to answer the correct answer without any explanation.Input: Question: {question}."
    return prompt

# 处理输入文件并生成结果
with open(f'{FILE_PATH}', 'r', encoding="utf-8") as f, open(f'{RESULT_FILE_PATH}llama_output.jsonl', 'w+', encoding="utf-8") as fout:
    all_data = json.load(f)
    total_count = len(all_data)
    for idx, data in enumerate(all_data, 1):
        question = data['question']
        question_id = data['question_id']
        options = data.get('options', [])
        input_images = data['input_image']
        answer = data.get('answer', '')
        category = data.get('category', '')
        answer_type = data.get('answer_type', '')

        # 多图处理
        if isinstance(input_images, list):
            image_names = input_images
        else:
            image_names = [input_images]
        image_filepaths = [os.path.join(IMAGE_DIR, name) for name in image_names]
        images = [Image.open(fp).convert('RGB').resize((224, 224)) for fp in image_filepaths]

        # 构造 prompt
        prompt = make_prompt(question, answer_type, options)

        # 构造 messages
        messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": fp} for fp in image_filepaths] +
                    [{"type": "text", "text": prompt}]
                )
            }
        ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

        # processor 输入
        inputs = processor(
            images,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)

        # 生成输出
        generated_id = model.generate(**inputs, max_new_tokens=30)
        full_output = processor.decode(generated_id[0]).strip()
        
        # 提取实际回答，只保留选项字母
        try:
            # 先从完整输出中提取助手的回答部分
            assistant_part = full_output.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            raw_answer = assistant_part.split("<|eot_id|>")[0].strip()
            if answer_type == 'choice':
                # 选择题，提取选项字母
                option_match = re.search(r'(?:^|\\s|\\n)([A-D])(?:[\\.。]|\\s|$)', raw_answer, re.IGNORECASE)
                if option_match:
                    output = option_match.group(1).upper()
                else:
                    text_option_match = re.search(r'(?:answer is|correct answer is|选择|选项)\\s*([A-D])', raw_answer, re.IGNORECASE)
                    if text_option_match:
                        output = text_option_match.group(1).upper()
                    else:
                        output = raw_answer
            else:
                # 非选择题，直接转大写
                output = raw_answer.upper()
        except:
            output = full_output.upper() if answer_type != 'choice' else full_output
        
        print(f"正在处理第{idx}/{total_count}个，ID: {question_id}")
        print(output)
        count += 1
        
        # 处理结果并写入输出文件
        if len(output) == 0:
            output = '--'

        if answer_type == 'choice':
            # 选择题：宽松包含判分
            if output.upper() in answer:
                result = 1
            elif answer in output.upper():
                result = 1
            else:
                result = 0
        else:
            # 非选择题：完全一致（不区分大小写）
            if output.strip().upper() == answer.strip().upper():
                result = 1
            else:
                result = 0

        result_json = {
            'id': question_id,
            'image': image_names[0],
            'result': result,
            'output': output,
            'answer': answer,
            'category': category
        }
        fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
        if result == 1:
            right_count += 1

        print(f"当前累计准确率: {right_count}/{count} = {right_count/count if count else 0:.4f}")

# 打印正确率
if count > 0:
    accuracy = right_count / count
    print(f"准确率: {accuracy:.2f} ({right_count}/{count})")
    

