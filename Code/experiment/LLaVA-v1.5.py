import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只用物理0号卡（你可以改成"3"）
import torch
import json
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images
from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token

def load_llava_model(model_path):
    print("Loading LLaVA model...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()
    # vision_tower device和精度
    if hasattr(model, "get_vision_tower"):
        vt = model.get_vision_tower()
        if vt is not None:
            vt.to(device)
            vt.half()
            if hasattr(vt, "vision_tower") and vt.vision_tower is not None:
                vt.vision_tower.to(device)
                vt.vision_tower.half()
    print(model)
    return tokenizer, model, image_processor

if __name__ == "__main__":
    DEVICE_INDEX = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    FILE_PATH = "xxx"
    RESULT_FILE_PATH = "xxx"
    IMAGE_DIR = "xxx"
    model_path = "xxx"

    # 加载 LLaVA 模型
    tokenizer, model, image_processor = load_llava_model(model_path)

    count = 0
    right_count = 0

    with open(FILE_PATH, 'r', encoding='utf-8') as f, \
         open(RESULT_FILE_PATH, 'w', encoding='utf-8') as fout:
        data_list = json.load(f)
        total = len(data_list)
        for idx, data in enumerate(data_list, 1):
            question = data['question']
            id = data['question_id']
            options = data.get('options', [])
            image_names = data['input_image']
            image_filepaths = [os.path.join(IMAGE_DIR, name) for name in image_names]
            images = [Image.open(path).convert("RGB") for path in image_filepaths]
            image_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
            image_sizes = [img.size for img in images]

            # 构造 prompt（deepseek风格）
            if options:
                options_str = ", Options: " + "; ".join(options)
            else:
                options_str = ""
            answer_type = data.get('answer_type', 'choice')
            if answer_type == 'choice' and options_str:
                prompt_content = f'You are currently an expert in a wide range of early childhood education topics as well as fundamental visual reasoning themes.Given the Image, a Question, your task is to answer the correct answer without any explanation. Note that you only need to choose one option from all options without explaining any reason. Input: Question: {question}{options_str}. \nOutput:'
            else:
                prompt_content = f'You are currently an expert in a wide range of early childhood education topics as well as fundamental visual reasoning themes.Given the Image, a Question, your task is to answer the correct answer without any explanation. Input: Question: {question}. \nOutput:'

            # tokenizer_image_token
            input_ids = tokenizer_image_token(prompt_content, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

            # 推理
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=50,
                    use_cache=True,
                )
            output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print(output)
            count += 1

            if len(output) == 0:
                output = '--'
            if answer_type == 'choice':
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
                if output.upper() == data['answer']:
                    result_json = {'id': id, 'image': data['input_image'][0], 'result': 1, 'output': output.upper(), 'answer': data['answer'], 'category': data['category']}
                    fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                    right_count += 1
                else:
                    result_json = {'id': id, 'image': data['input_image'][0], 'result': 0, 'output': output.upper(), 'answer': data['answer'], 'category': data['category']}
                    fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')

            print(f"进度: {idx}/{total}，当前准确率: {right_count/count:.2%} ({right_count}/{count})")

    if count > 0:
        accuracy = right_count / count
        print(f"准确率: {accuracy:.2f} ({right_count}/{count})")
