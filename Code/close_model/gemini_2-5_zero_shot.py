import requests
import json
from io import BytesIO
import PIL.Image as Image
import base64

# API配置
API_KEY = "xxx"
API_URL = "xxx"

# 图片目录设置
IMAGE_DIR = "xxx"
count = 0
right_count = 0

def fetch_image_content(image_path):
    if image_path.startswith(('http://', 'https://')):
        # 处理网络图片
        response = requests.get(image_path)
        if response.status_code == 200:
            return BytesIO(response.content)
        return None
    else:
        # 处理本地文件
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
        # 如果图像有透明通道，先填充白色背景
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # 使用Alpha通道作为掩码
            image = background

        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return 'data:image/jpeg;base64,' + img_str
    except Exception as e:
        print(f"encoding error: {e}")
        return None

def call_gemini(prompt: str, images):
    # 准备消息内容
    content = [{"type": "text", "text": prompt}]
    for image in images:
        if image:  # 只添加存在的图片
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image
                }
            })

    data = {
        "model": "gemini-2.5-pro-exp-03-25",
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }

    headers = {
        "Authorization": API_KEY
    }

    try:
        response = requests.post(API_URL, json=data, headers=headers)
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text}")
        
        response.raise_for_status()
        response_json = response.json()
        return response_json['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error during API call: {e}")
        return None

def build_prompt(question_data):
    """根据题型构造英文提示词"""
    base =  f'You are currently an expert in a wide range of early childhood education topics as well as fundamental visual reasoning themes.' \
            f'\nGiven one or more Images, a Question and Options, your task is to answer the question correctly. ' \
            f'\nOnly answer each question without explaining.' \

    if question_data['answer_type'] == 'choice':
        return f"""{base}
        Question: {question_data['question']}
        Options: {"; ".join(question_data['options'])}"""

    elif question_data['answer_type'] == 'blank':
        return f"""{base}
        Question: {question_data['question']}"""

    elif question_data['answer_type'] == 'order':
        return f"""{base}
        Question: {question_data['question']}"""

    return f"{base}\nQuestion: {question_data['question']}"

def process_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    with open(output_file, 'w', encoding='utf-8') as out_file:
        for item in data:
            question_id = item['question_id']
            question = item['question']
            category = item['category']
            input_images = item['input_image']
            if isinstance(input_images, str):
                input_images = [input_images]  # 统一成列表
            answer_type = item['answer_type']
            answer = item['answer']

            # 使用build_prompt函数构造提示词
            prompt = build_prompt(item)

            # 处理所有图片
            encoded_images = []
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

            # 尝试调用Gemini API
            model_answer = None
            for _ in range(3):
                try:
                    model_answer = call_gemini(prompt, encoded_images)
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

            # 处理答案
            output = model_answer.strip().rstrip('.').upper()

            # 处理选择题答案
            if answer_type == 'choice':
                option_match = None
                if output:
                    for char in output:
                        if char.isalpha() and char.upper() in ['A', 'B', 'C', 'D', 'E']:
                            option_match = char.upper()
                            break
                output = option_match if option_match else "INVALID"
                is_correct = 1 if output == answer.upper() else 0
            else:
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
            print(f'Model output: {output}')
            print(f'Correct answer: {answer}')
            print(f'Current accuracy: {right_count}/{count} = {right_count / count if count else 0:.2f}')
            print('---')

    accuracy = right_count / count if count else 0
    print(f'Final accuracy: {accuracy:.3f}')

input_file_path = "xxx"  # 输入文件
output_file_path = "xxx"  # 输出文件

process_json(input_file_path, output_file_path)