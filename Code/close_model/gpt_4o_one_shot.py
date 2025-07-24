# import openai
from openai import OpenAI
import json
import requests
from io import BytesIO
import PIL.Image as Image
import base64
import random
import os

client = OpenAI(
    api_key='xxx',
    base_url='xxx'
)

# Modify this to your local image directory or URL base
IMAGE_DIR = 'xxx'
count = 0
right_count = 0

# 预加载所有category的示例数据
CATEGORY_EXAMPLES_DIR = 'xxx'
category_examples = {}

for fname in os.listdir(CATEGORY_EXAMPLES_DIR):
    if fname.endswith('.json'):
        cat = fname[:-5]
        with open(os.path.join(CATEGORY_EXAMPLES_DIR, fname), 'r', encoding='utf-8') as f:
            items = json.load(f)
            # 按answer_type分组
            ans_type_dict = {}
            for item in items:
                ans_type = item['answer_type']
                ans_type_dict.setdefault(ans_type, []).append(item)
            category_examples[cat] = ans_type_dict

def fetch_image_content(image_path):
    if image_path.startswith(('http://', 'https://')):
        response = requests.get(image_path)
        if response.status_code == 200:
            return BytesIO(response.content)
        return None
    else:
        try:
            with open(image_path, 'rb') as f:
                return BytesIO(f.read())
        except FileNotFoundError:
            print(f"文件未找到: {image_path}")
            return None

def encode_image(image):
    if image is None:
        return None
    buffered = BytesIO()
    try:
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return 'data:image/jpeg;base64,' + img_str
    except Exception as e:
        print(f"encoding error: {e}")
        return None

def call_gpt4(prompt: str, images, detail='auto'):
    try:
        content = [{"type": "text", "text": prompt}]
        for image in images:
            if image:
                content.append({"type": "image_url", "image_url": {"url": image, "detail": detail}})
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=500,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during answering: {e}")
        return None

def get_oneshot_example(category, answer_type):
    """从category_examples中随机选取一条同answer_type的示例"""
    if category in category_examples and answer_type in category_examples[category]:
        return random.choice(category_examples[category][answer_type])
    else:
        return None

def get_oneshot_prompt(example, question_type, question, options=None):
    # example为选中的one-shot示例
    base_prompt = (
        'You are currently an expert in a wide range of early childhood education topics as well as fundamental visual reasoning themes.'
        '\nGiven one or more Images and a Question, your task is to answer the question correctly.'
        '\nOnly answer each question without explaining.'
        '\nExample:'
        f'\nQuestion: {example["question"]}'
    )
    if question_type == 'choice':
        base_prompt += f'\nOptions: {"; ".join(example["options"])}'
        base_prompt += f'\n{example["answer"]}'
        base_prompt += f'\n---\nQuestion: {question}\nOptions: {"; ".join(options) if options else "N/A"}'
    else:
        base_prompt += f'\n{example["answer"]}'
        base_prompt += f'\n---\nQuestion: {question}'
    return base_prompt

def process_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for item in data:
            question_id = item['question_id']
            question = item['question']
            category = item['category']
            input_images = item['input_image']
            answer_type = item['answer_type']
            answer = item['answer']
            
            # 选取one-shot示例
            oneshot_example = get_oneshot_example(category, answer_type)
            if not oneshot_example:
                print(f"未找到category={category} answer_type={answer_type}的one-shot示例，跳过")
                continue
            oneshot_prompt = get_oneshot_prompt(
                oneshot_example,
                answer_type,
                question,
                item.get('options') if answer_type == 'choice' else None
            )
            
            # 处理所有图片（one-shot图片+当前问题图片）
            encoded_images = []
            
            # one-shot图片（支持多图）
            if oneshot_example['input_image']:
                for oneshot_img_name in oneshot_example['input_image']:
                    oneshot_image_path = f'{IMAGE_DIR}/{oneshot_img_name}'
                    oneshot_image_content = fetch_image_content(oneshot_image_path)
                    if oneshot_image_content is not None:
                        try:
                            oneshot_image = Image.open(oneshot_image_content)
                            oneshot_image_encoded = encode_image(oneshot_image)
                            if oneshot_image_encoded:
                                encoded_images.append(oneshot_image_encoded)
                        except Exception as e:
                            print(f"读取one-shot图片出错: {oneshot_img_name}, 错误: {e}")
                    else:
                        print(f"Warning: One-shot example image not found: {oneshot_img_name}")
            else:
                print("Warning: One-shot example has no input_image.")
            
            # 当前问题图片
            for img_name in input_images:
                image_path = f'{IMAGE_DIR}/{img_name}'
                image_content = fetch_image_content(image_path)
                if image_content is not None:
                    image = Image.open(image_content)
                    image_encoded = encode_image(image)
                    if image_encoded:
                        encoded_images.append(image_encoded)
            
            global count
            global right_count
            if not encoded_images:
                print(f"No valid images found for question {question_id}")
                result_json = {
                    "question_id": question_id,
                    "result": 0,
                    "output": "NO_IMAGE",
                    "answer": answer,
                    "category": category,
                    "first_image": input_images[0] if input_images else "NO_IMAGE"
                }
                out_file.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                count += 1
                continue
            
            model_answer = None
            for _ in range(3):
                try:
                    model_answer = call_gpt4(oneshot_prompt, encoded_images)
                    if model_answer:
                        break
                except Exception as e:
                    print(f"Attempt failed for question {question_id}: {e}")
                    continue
            
            if not model_answer:
                result_json = {
                    "question_id": question_id,
                    "result": 0,
                    "output": "API_ERROR",
                    "answer": answer,
                    "category": category,
                    "first_image": input_images[0] if input_images else "NO_IMAGE"
                }
                out_file.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                count += 1
                continue
            
            output = model_answer.strip().rstrip('.').upper()
            
            # 根据不同题型处理答案
            if answer_type == 'choice':
                option_match = None
                if output:
                    for char in output:
                        if char.isalpha() and char.upper() in ['A', 'B', 'C', 'D', 'E']:
                            option_match = char.upper()
                            break
                output = option_match if option_match else "INVALID"
                is_correct = 1 if output == answer.upper() else 0
            elif answer_type == 'order':
                # 对于排序题，标准化输出格式（移除空格，转换为小写）
                output = ''.join(output.split()).lower()
                answer = ''.join(answer.split()).lower()
                is_correct = 1 if output == answer else 0
            else:  # blank
                is_correct = 1 if output.lower() == answer.lower() else 0
            
            result_json = {
                "question_id": question_id,
                "result": is_correct,
                "output": output,
                "answer": answer,
                "category": category,
                "first_image": input_images[0] if input_images else "NO_IMAGE"
            }
            out_file.write(json.dumps(result_json, ensure_ascii=False) + '\n')
            
            count += 1
            if is_correct:
                right_count += 1
            
            print(f'Question ID: {question_id}')
            print(f'One-shot Example ID: {oneshot_example.get("question_id", "N/A")}')
            print(f'Model output: {output}')
            print(f'Correct answer: {answer}')
            print(f'Current accuracy: {right_count}/{count} = {right_count / count if count else 0:.2f}')
            print('---')
    
    accuracy = right_count / count if count else 0
    print(f'Final accuracy: {accuracy:.3f}')

input_file_path = "xxx"  # 输入文件
output_file_path = "xxx"  # 输出文件

process_json(input_file_path, output_file_path) 