import json
from PIL import Image
import torch
import os
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

class EnhancedEvaluationTracker:
    """增强版评估跟踪器，带实时结果显示和详细统计"""
    def __init__(self):
        self.total = 0
        self.correct = 0
        self.history = []
    
    def update(self, question_data, output, is_correct):
        """更新统计并记录历史"""
        self.total += 1
        self.correct += int(is_correct) if is_correct is not None else 0
        
        # 记录详细结果（包含时间戳）
        result_entry = {
            "question_id": question_data["question_id"],
            "question": question_data["question"],
            "output": output,
            "correct": is_correct,
            "options": question_data.get("options", []),
            "ground_truth": question_data["answer"],
            "category": question_data.get("category", "")
        }
        self.history.append(result_entry)
        self._print_realtime_result(result_entry)
    
    @property
    def accuracy(self):
        return self.correct / max(1, self.total)
    
    def _print_realtime_result(self, result):
        """打印实时结果（兼容普通终端）"""
        print(f"\n[Q{result['question_id']}]")
        print(f"问题: {result['question']}")
        if result["options"]:
            print("选项:", " | ".join(result["options"]))
        print(f"模型输出: {result['output']}")
        print(f"正确答案: {result['ground_truth']}")
        status = "✓ 正确" if result["correct"] else "✗ 错误"
        print(f"结果: {status}")
        print(f"累计准确率: {self.accuracy:.2%} ({self.correct}/{self.total})")
        print("-" * 60)
    
    def get_summary(self):
        """获取最终统计摘要"""
        return {
            "total_questions": self.total,
            "correct_answers": self.correct,
            "accuracy": f"{self.accuracy:.2%}",
            "incorrect_ids": [item["question_id"] for item in self.history if not item["correct"]]
        }
def load_model_and_processor(model_path="xxx"):
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    ).half()
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    processor.image_processor.size = {"shortest_edge": 448}
    return model, processor

def evaluate_question(question_data, model, processor, tracker, image_base_path=""):
    """评估单个问题（增强版）"""
    # 构建对话消息
    messages = [
        {"role": "system", "content": "You are currently an expert in a wide range of early childhood education topics as well as fundamental visual reasoning themes.Given the Image, a Question, your task is to answer the correct answer without any explanation."},
        {"role": "user", "content": []}
    ]
    
    # 添加图片
    for img_path in question_data.get("input_image", []):
        full_path = f"{image_base_path}/{img_path}"
        messages[1]["content"].append({"type": "image", "image": f"file://{full_path}"})
    
    # 根据题型定制提示词
    question_text = question_data["question"]
    if question_data["answer_type"] == "choice":
        # 拼接选项内容
        options = question_data.get("options", [])
        if options:
            options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            question_text += "\n" + options_str
        question_text += "\nNote that you only need to choose one option from all options without explaining any reason."
    
    messages[1]["content"].append({"type": "text", "text": question_text})
    
    try:
        # 准备模型输入
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        # 生成输出（严格限制输出长度）
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=10,  # 更短的输出限制
            pad_token_id=processor.tokenizer.pad_token_id
        )
        
        # 解码并后处理输出
        output = processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()
        
        # 验证输出有效性
        processed_output = postprocess_output(output, question_data)
        is_correct = validate_answer(processed_output, question_data)
        
        # 更新跟踪器
        tracker.update(question_data, processed_output, is_correct)
        
        return {
            "question_id": question_data["question_id"],
            "output": processed_output,  # 始终记录模型的回答
            "ground_truth": question_data["answer"],
            "result": is_correct,
            "category": question_data.get("category", "")
        }
        
    except Exception as e:
        print(f"\n处理问题 {question_data['question_id']} 时出错: {str(e)}")
        tracker.update(question_data, "ERROR", None)
        return {
            "question_id": question_data["question_id"],
            "error": str(e),
            "result": None
        }

def postprocess_output(output, question_data):
    """增强型输出后处理"""
    output = output.strip().upper()
    
    if question_data["answer_type"] == "choice":
        # 提取第一个有效选项字母
        for char in output:
            if char in ['A', 'B', 'C', 'D', 'E', 'F']:
                return char
        return output[0] if output else ""
    
    else:
        # 去除多余符号
        return output.split()[0].strip(",.!?;:\"'")

    return output

def validate_answer(output, question_data):
    """验证答案正确性（支持多种题型）"""
    if question_data["answer_type"] == "choice":
        return output in question_data["answer"].upper()
    else:
        return str(output) == str(question_data["answer"])

    return False

def evaluate_all_questions(test_data_path, output_path, image_base_path="", start_question_id=None):
    """批量评估主函数（支持从指定question_id开始，续写结果）"""
    # 加载模型
    model, processor = load_model_and_processor()
    tracker = EnhancedEvaluationTracker()
    
    # 读取测试数据
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 确定开始位置（支持续写）
    start_index = 0
    existing_results = []
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_results = [json.loads(line) for line in f if line.strip()]
        
        if start_question_id:
            # 优先使用指定的起始ID
            start_index = next((i for i, q in enumerate(test_data) 
                              if q["question_id"] == start_question_id), 0)
        elif existing_results:
            # 自动从上次的最后一个问题继续
            last_id = existing_results[-1]["question_id"]
            start_index = next((i for i, q in enumerate(test_data) 
                              if q["question_id"] == last_id), 0) + 1
    
    # 打开文件（追加模式）
    with open(output_path, 'a' if existing_results else 'w', encoding='utf-8') as fout:
        # 评估过程（从指定位置开始）
        for question in tqdm(test_data[start_index:], 
                           desc=f"评估进度（从ID:{test_data[start_index]['question_id']}）",
                           initial=start_index,
                           total=len(test_data)):
            
            # 跳过已处理的问题（确保幂等性）
            if any(r["question_id"] == question["question_id"] for r in existing_results):
                continue
                
            eval_result = evaluate_question(question, model, processor, tracker, image_base_path)
            
            # 构建输出记录
            record = {
                "question_id": question["question_id"],
                "question": question["question"],
                "output": eval_result.get("output", "ERROR"),
                "category": question.get("category", "unknown"),
                "ground_truth": question["answer"],
                "result": 1 if eval_result["result"] else 0,
                
            }
            
            # 写入文件（确保立即刷新）
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
            fout.flush()
    
    # 打印报告
    print("\n" + "="*60)
    print(f"评估完成！最终准确率: {tracker.accuracy:.2%}")
    print(f"最后处理的题目ID: {test_data[-1]['question_id']}")
    print("="*60)
# 使用示例
if __name__ == "__main__":

    evaluate_all_questions(
        test_data_path="xxx",
        output_path="xxx",
        image_base_path="xxx"
    )