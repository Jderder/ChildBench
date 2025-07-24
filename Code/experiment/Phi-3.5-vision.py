import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
import torch
import re
from PIL import Image 
from transformers import AutoModelForCausalLM, AutoProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FILE_PATH = "xxx"  # 输入文件路径（标准json格式）
RESULT_FILE_PATH = "xxx"  # 输出文件路径
IMAGE_DIR = "xxx"  # 图片目录
model_id = "xxx"

# Note: set _attn_implementation='eager' if you don't have flash_attn installed
model = AutoModelForCausalLM.from_pretrained(
  model_id, 
  device_map="auto",  # 这里的0是指当前可见的第一个GPU（即CUDA_VISIBLE_DEVICES=2时的2号卡）
  trust_remote_code=True, 
  torch_dtype="auto", 
  _attn_implementation='eager'    
)

# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
processor = AutoProcessor.from_pretrained(model_id, 
  trust_remote_code=True, 
  num_crops=4  # 单帧图像使用4
) 

def make_prompt(question, answer_type, options=None):
    options_str = ""
    if options:
        options_str = " Options: " + " ".join(options)
    if answer_type == 'choice' and options_str:
        prompt = f'You are currently an expert in a wide range of early childhood education topics as well as fundamental visual reasoning themes.Given the Image, a Question, your task is to answer the correct answer without any explanation. Note that you only need to choose one option from all options without explaining any reason. Input: Question: {question}{options_str}. \nOutput:'
    else:
        prompt = f'You are currently an expert in a wide range of early childhood education topics as well as fundamental visual reasoning themes.Given the Image, a Question, your task is to answer the correct answer without any explanation. Input: Question: {question}. \nOutput:'
    return prompt

def make_image_placeholders(num_images):
    return ''.join([f"<|image_{i+1}|>\n" for i in range(num_images)])

def add_image_placeholders(question, num_images):
    placeholders = '\n'.join(['<image>'] * num_images)
    return f"{placeholders}\n{question}"

def replace_image_placeholders(question, num_images):
    # 统一替换所有 <image>、<image_1>、<image_2>...为 <image>
    pattern = re.compile(r'<image(?:_\d+)?>')
    matches = list(pattern.finditer(question))
    if len(matches) != num_images:
        print(f"警告：图片数与占位符数不一致！")
    # 替换为 <image>，并确保前后有换行或空格
    new_question = pattern.sub(' <image> ', question)
    # 再把多个空格合并为一个
    new_question = re.sub(r'\s+', ' ', new_question)
    return new_question.strip()

def make_image_placeholders_and_options(question, options, num_images):
    question_with_first_image = f"<|image_1|>\n{question}"
    options_str = ""
    for i, opt in enumerate(options):
        if i+2 <= num_images:
            options_str += f"\n{chr(65+i)}.<|image_{i+2}|>{opt[2:]}"
        else:
            options_str += f"\n{opt}"
    return question_with_first_image, options_str

count = 0
right_count = 0

with open(FILE_PATH, 'r', encoding="utf-8") as f, open(RESULT_FILE_PATH, 'w+', encoding="utf-8") as fout:
    all_data = json.load(f)
    for idx, item in enumerate(all_data, 1):
        question_id = item["question_id"]
        question = item["question"]
        answer_type = item.get("answer_type", "")
        options = item.get("options", None)
        input_images = item["input_image"]
        answer = item.get("answer", "")
        category = item.get("category", "")

        if isinstance(input_images, list):
            image_names = input_images
        else:
            image_names = [input_images]
        image_filepaths = [os.path.join(IMAGE_DIR, name) for name in image_names]
        images = [Image.open(fp).convert('RGB').resize((224, 224)) for fp in image_filepaths]
        # print(len(images))

        if answer_type == 'choice' and options:
            question_with_first_image, options_str = make_image_placeholders_and_options(question, options, len(images))
            prompt = f"Given the Image, a Question, your task is to answer the correct answer without any explanation. Note that you only need to choose one option from all options without explaining any reason. Input: Question: {question_with_first_image}{options_str}. \nOutput:"
        else:
            image_placeholders = ''.join([f"<|image_{i+1}|>\n" for i in range(len(images))])
            question_with_placeholders = image_placeholders + question
            prompt = make_prompt(question_with_placeholders, answer_type, options)
        # print("prompt内容：", prompt)
        # print("图片数：", len(images))
        messages = [{"role": "user", "content": prompt}]

        inputs = processor(text=prompt, images=images, return_tensors="pt").to("cuda:0") 

        generation_args = { 
            "max_new_tokens": 50, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 

        generate_ids = model.generate(**inputs, 
            eos_token_id=processor.tokenizer.eos_token_id, 
            **generation_args
        )

        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

        print(output)
        count += 1

        if len(output) == 0:
            output = '--'
        if answer_type == 'choice':
            if output.upper() in answer:
                result = 1
            elif answer in output.upper():
                result = 1
            else:
                result = 0
        else:
            if output.strip().upper() == answer.strip().upper():
                result = 1
            else:
                result = 0
        if result == 1:
            right_count += 1

        result_json = {
            'id': question_id,
            'image': image_names[0],
            'result': result,
            'output': output,
            'answer': answer,
            'category': category
        }
        fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')

        print(f"正在处理第{idx}/{len(all_data)}个，ID: {question_id}")
        print(f"当前累计准确率: {right_count}/{count} = {right_count/count if count else 0:.4f}")

        # 推理后清理显存
        torch.cuda.empty_cache()

if count > 0:
    accuracy = right_count / count
    print(f"准确率: {accuracy:.2f} ({right_count}/{count})")
    

