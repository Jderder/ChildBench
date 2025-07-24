import json
import re
import sys

def is_valid_option(s):
    return isinstance(s, str) and re.fullmatch(r'[A-D]', s)

# 读取并筛选选择题数据，并将output转为大写
input_file = "xxx"
if len(sys.argv) > 1:
    input_file = sys.argv[1]
data = []
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"警告：第{line_num}行JSON解析错误: {e}")
                continue
except FileNotFoundError:
    print(f"错误：找不到文件 {input_file}")
    sys.exit(1)
except Exception as e:
    print(f"错误：读取文件时发生错误: {e}")
    sys.exit(1)

if not data:
    print("错误：没有成功读取到任何数据")
    sys.exit(1)

choice_data = []
for d in data:
    output_upper = d['output'].upper() if isinstance(d['output'], str) else d['output']
    answer = d['answer']
    if is_valid_option(output_upper) and is_valid_option(answer):
        d = d.copy()
        d['output'] = output_upper
        choice_data.append(d)

if not choice_data:
    print("警告：没有找到符合条件的A-D选项数据")
    sys.exit(0)

options = ['A', 'B', 'C', 'D']

# 初始化统计字典
stats = {opt: {'TP': 0, 'FP': 0, 'FN': 0} for opt in options}

# 统计每个选项的TP、FP、FN
for d in choice_data:
    pred = d['output']
    gold = d['answer']
    for opt in options:
        if pred == opt and gold == opt:
            stats[opt]['TP'] += 1
        elif pred == opt and gold != opt:
            stats[opt]['FP'] += 1
        elif pred != opt and gold == opt:
            stats[opt]['FN'] += 1

# 计算每个选项的P、R、F1并存储
print("各选项的精确率、召回率和F1值：")
all_options_metrics = {}
for opt in options:
    TP = stats[opt]['TP']
    FP = stats[opt]['FP']
    FN = stats[opt]['FN']
    P = TP / (TP + FP) if (TP + FP) > 0 else 0
    R = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
    all_options_metrics[opt] = {'P': P, 'R': R, 'F1': F1}
    print(f'选项{opt}: P={P:.4f}, R={R:.4f}, F1={F1:.4f} (TP={TP}, FP={FP}, FN={FN})')

# 计算宏平均
macro_P = sum(m['P'] for m in all_options_metrics.values()) / len(options)
macro_R = sum(m['R'] for m in all_options_metrics.values()) / len(options)
macro_F1 = sum(m['F1'] for m in all_options_metrics.values()) / len(options)

print(f'\n宏平均: P={macro_P:.4f}, R={macro_R:.4f}, F1={macro_F1:.4f}')
