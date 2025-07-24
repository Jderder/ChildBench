import torch
import json
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
from PIL import Image
import os

DEVICE_INDEX = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FILE_PATH = "xxx"  # 输入文件路径
RESULT_FILE_PATH = "xxx"  # 输出文件路径
IMAGE_DIR = "xxx"  # 图片目录

# 加载模型
model_path = "xxx"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(DEVICE_INDEX).to(torch.bfloat16).cuda().eval()

# 初始化计数器
count = 0
right_count = 0

# 处理输入文件并生成结果
with open('xxx', 'r', encoding='utf-8') as f, \
     open('xxx', 'w', encoding='utf-8') as fout:
    data_list = json.load(f)
    total = len(data_list)
    for idx, data in enumerate(data_list, 1):
        question = data['question']
        id = data['question_id']
        options = data.get('options', [])
        image_names = data['input_image']
        image_filepaths = [os.path.join(IMAGE_DIR, name) for name in image_names]
        pil_images = [Image.open(path) for path in image_filepaths]
        
        # 构造 prompt
        if options:
            options_str = ", Options: " + "; ".join(options)
        else:
            options_str = ""
        # 判断是否为填空题
        if options_str:
            prompt_content = f'You are currently an expert in a wide range of early childhood education topics as well as fundamental visual reasoning themes.Given the Image, a Question, your task is to answer the correct answer without any explanation. Note that you only need to choose one option from the all options without explaining any reason. Input: Question: {question}{options_str}. \nOutput:'
        else:
            prompt_content = f'You are currently an expert in a wide range of early childhood education topics as well as fundamental visual reasoning themes.Given the Image, a Question, your task is to answer the correct answer without any explanation. Input: Question: {question}. \nOutput:'

        conversation = [
            {
                "role": "User",
                "content": prompt_content,
                "images": image_filepaths
            }
        ]

        # # 打印给模型的输入内容
        # print("[DEBUG] conversation 输入:")
        # print(json.dumps(conversation, ensure_ascii=False, indent=2))

        # 加载图片并准备输入
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(vl_gpt.device)

        # 运行图像编码器获取图像嵌入
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # 运行模型获取响应
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=256,
            do_sample=False,
            use_cache=True
        )

        # 解码输出
        generated_text = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).strip().rstrip('.')
        print(generated_text)
        output = generated_text
        count += 1
        
        # 处理结果并写入输出文件
        answer_type = data.get('answer_type', 'choice')
        if len(output) == 0:
            output = '--'
        if answer_type == 'choice':
            # 选择题，宽松判分
            if output.upper() in data['answer']:
                result_json = {'id': id, 'image': data['input_image'][0], 'result': 1, 'output': output.upper(), 'answer': data['answer'], 'category': data['category']}
                fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                right_count += 1
            elif data['answer'] in output.upper():
                result_json = {'id': id, 'image': data['input_image'][0], 'result': 1, 'output': output.upper(), 'answer': data['answer'], 'category': data['category']}
                fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                right_count += 1
            else:
                result_json = {'id': id, 'image': data['input_image'][0], 'result': 0, 'output': output.upper(), 'answer': data['answer'], 'category': data['category']}
                fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
        else:
            # 其它题型，严格判分
            if output.upper() == data['answer']:
                result_json = {'id': id, 'image': data['input_image'][0], 'result': 1, 'output': output.upper(), 'answer': data['answer'], 'category': data['category']}
                fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                right_count += 1
            else:
                result_json = {'id': id, 'image': data['input_image'][0], 'result': 0, 'output': output.upper(), 'answer': data['answer'], 'category': data['category']}
                fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')

        # 实时打印进度和准确率
        print(f"进度: {idx}/{total}，当前准确率: {right_count/count:.2%} ({right_count}/{count})")

# 打印正确率
if count > 0:
    accuracy = right_count / count
    print(f"准确率: {accuracy:.2f} ({right_count}/{count})")
