# import openai
from openai import OpenAI
import json
import requests
from io import BytesIO
import PIL.Image as Image
import base64

client = OpenAI(
    api_key='xxx',
    base_url='xxx'
)

# Modify this to your local image directory or URL base
IMAGE_DIR = 'xxx'
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


def call_gpt4(prompt: str, images, detail='auto'):
    try:
        # Prepare the message content with text and all images
        content = [{"type": "text", "text": prompt}]
        for image in images:
            if image:  # Only add if image exists
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

            # Handle different question types
            if answer_type == 'choice':
                options = item['options']
                prompt = f'You are currently an expert in a wide range of early childhood education topics as well as fundamental visual reasoning themes.' \
                         f'\nGiven one or more Images, a Question and Options, your task is to answer the question correctly. ' \
                         f'\nOnly answer each question without explaining.' \
                         f'\nQuestion: {question}' \
                         f'\nOptions: {"; ".join(options) if options else "N/A"}' \

            else:
                prompt = f'You are currently an expert in a wide range of early childhood education topics as well as fundamental visual reasoning themes.' \
                         f'\nGiven one or more Images and a Question, your task is to answer the question correctly. ' \
                         f'\nProvide only the answer without any explanation.' \
                         f'\nQuestion: {question}' \


            # Process all images for this question
            encoded_images = []
            for img_name in input_images:
                # Modify this according to how your images are stored
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

            # Try calling GPT-4 up to 3 times
            model_answer = None
            for _ in range(3):
                try:
                    model_answer = call_gpt4(prompt, encoded_images)
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

            # Process the answer based on answer type
            output = model_answer.strip().rstrip('.').upper()  # Standardize to uppercase

            # For multiple-choice questions, extract just the letter
            if answer_type == 'choice':
                # Extract the first capital letter from the response
                option_match = None
                if output:
                    for char in output:
                        if char.isalpha() and char.upper() in ['A', 'B', 'C', 'D', 'E']:
                            option_match = char.upper()
                            break
                output = option_match if option_match else "INVALID"

                # Compare with answer (which is just the letter in the JSON)
                is_correct = 1 if output == answer.upper() else 0
            else:
                # For blank/order questions, do direct comparison (case insensitive)
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


input_file_path = "xxx"  # Your new JSON file
output_file_path = "xxx"  # Output file

process_json(input_file_path, output_file_path)