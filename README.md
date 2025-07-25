<br />
<p align="center">
  <h1 align="center">✏️Easy for Children, Hard for AI: The Limits of MLLMs in Early Learning Tasks
Recognition </h1>
  <h3 align="center">ChildBench: A new benchmark dataset for Early education of MLLMs.</h3>
  
  <p align="center">  
<!--     <a href="">arxiv</a> -->
    ·
    <a href="https://anonymous.4open.science/r/ChildBench-AF78/Dateset/dataset/train.jsonl">github</a>
    ·
    <a href="https://anonymous.4open.science/r/ChildBench-AF78/LICENSE">license</a>
<!--     <a href="">benchmark</a> -->
    
  </p>
</p>


## Contents

- [ChildBench](#Contents)
  - [Overview](#1-Overview)
    - [Examples](#Examples)
    - [Detail Information](#Detail-Information)
  - [Access ChildBench](#2-Access-ChildBench)
    - [Data Split](#Data-Split)
    - [Data Format](#Data-Format)
  - [Experiment & Evaluation](#3-Experiment-and-Evaluation)
    - [Experiment](#Experiment)
    - [Evaluation](#Evaluation)
  - [License](#4-License)




## 1 Overview
**ChildBench** is a high-quality benchmark dataset that undergoes **manual image construction** and **manual annotation**, specifically designed to evaluate the core capabilities of multimodal large language models in early education scenarios. The dataset includes 4,890 images and 5,346 annotated samples, covering 10 task types across 5 capability assessments, including core capability dimensions such as spatial reasoning, counting, and visual tracking. To ensure data quality, ChildBench has formulated detailed construction and annotation processes, with a team of professional annotators and auditors conducting three rounds of rigorous quality control.
### Examples
The following figures list some classic examples in our dataset. You can click out [`Examples`](Examples) to view partial details of the dataset.

### Detail Information
The following table [`Splits/`](Dataset/type/splits.png) lists the detailed information statistics of the splited dataset.
<br>
You can find our dataset through the following path **_(Dataset/dataset)_** for more details.
<br>
_Due to the fact that only redirecting to the specified file is valid in anonymous links, redirecting to the specified directory is invalid. Therefore, we use bold and italicized font to indicate the markings of all specified directories, making it easier for reviewers to search. Thank you!_


## 2 Access ChildBench
All images in the dataset can be viewed after downloading the compressed package in the **_(Dataset/images)_** directory.

###  Data Split
As reported in the folloeing table, ChildBench contains 5,346 samples, divided into training, validation, and test sets according to a 7:1:2 ratio.
<br>All the splited data sets are in the directory **_(Dataset/dataset)_**. 


### Data Format
Each `json` file is of the following format:
```json
[
  {
    "question_id": "1",
    "question": "Determine rows 1-4 separately. Which shape in each row is different from the other shapes?",
    "category": "Observation_Skills-Spot_the_Differences",
    "input_image": [
      "Observation_Skills_Spot_the_Differences_0012.png"
    ],
    "answer_type": "choice",
    "options": [
      "A.A,B,A,D",
      "B.B,B,A,C",
      "C.A,B,A,C",
      "D.C,B,A,C"
    ],
    "answer": "A"
  },
  {
    "question_id": "2",
    "question": "What are the corresponding letters of the overlapping shapes in the picture from top to bottom?",
    "category": "Spatial_Reasoning-Overlap_Order",
    "input_image": [
      "Spatial_Reasoning_Overlap_Order_0252.png"
    ],
    "answer_type": "order",
    "answer": "A, E, C, D, B"
  },
  {
    "question_id": "3",
    "question": "What numbers correspond to the different shapes from top to bottom in the picture? ",
    "category": "Visual_Tracking",
    "input_image": [
      "Visual_Tracking_0009.png"
    ],
    "answer_type": "choice",
    "options": [
      "A.2,3,5,4,1",
      "B.2,4,5,3,1",
      "C.3,1,2,5,4",
      "D.4,2,5,3,1"
    ],
    "answer": "B"
  },
  {
    "question_id": "4",
    "question": "How many cubes are there in the picture?",
    "category": "Counting_Skills-Spatial_Counting",
    "input_image": [
      "Counting_Skills_Spatial_Counting_0084.png"
    ],
    "answer_type": "blank",
    "answer": "7"
  }
]
```
Each line is an individual data point.
 `question`  is the question with manual annotation, `input_image`  denotes name of the image, `options`  is reasonable numerical options.
<br>
It encompasses 5 capability evaluations across 10 task types, precisely aligning with early - education assessment scenarios.

## 3 Experiment and Evaluation
### Experiment
We have disclosed the inference code for the model in the directory **_(Code/experiment)_**,  as well as the fine-tuning code in the directory **_(Code/finetune)_**.

> Note: Before using any code or scripts in this project, you need to manually supplement necessary path information in the relevant files, including but not limited to model path, training file path, and output path. 
<br>

- For all 8 open-sourse MLLMs, you can directly execute Python files in the directory **_(Code/experiment)_** to perform inference on models before and after fine-tuning: 
```
nohup python DeepSeek-VL.py > log/DeepSeek-VL.log 2>1& &
nohup python InternVL3.py > log/InternVL3.log 2>1& &
nohup python Llama-3.2-Vision.py > log/Llama-3.2-Vision.log 2>1& &
nohup python LLaVA-v1.5.py > log/LLaVA-v1.6.log 2>1& &
nohup python MiniCPM-V-2.6.py > log/MiniCPM-V-2.6.log 2>1& &
nohup python mPLUG-Owl3.py > log/mPLUG-Owl3.log 2>1& &
nohup python Phi-3.5-vision.py > log/Phi-3.5-vision.log 2>1& &
nohup python Qwen2.5-VL.py > log/Qwen2.5-VL.log 2>1& &
```
Due to the large amount of open source model code, you need to download it yourself through channels or call it directly from platforms such as [huggingface](https://huggingface.co).
- For open-source models, You can execute Bash files using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [ms-swift](https://github.com/modelscope/ms-swift) in the directory **_(Code/finetune)_** to perform fine-tuning.For models without provided Bash scripts, you can directly use LLaMA Factory's webui for fine-tuning with default parameters:
```
nohup bash DeepSeek-VL.sh > log/DeepSeek-VL_train.log 2>1& &
nohup bash InternVL3.sh > log/InternVL3_train.log 2>1& &
nohup bash Llama-3.2-Vision.sh > log/Llama-3.2-Vision_train.log 2>1& &
nohup bash LLaVA-v1.5.sh > log/LLaVA-v1.6_train.log 2>1& &
nohup bash mPLUG-Owl3.sh > log/mPLUG-Owl3_train.log 2>1& &
nohup bash Phi-3.5-vision.sh > log/Phi-3.5-vision_train.log 2>1& &
```
- For gemini-2.5-pro and gpt-4o, you can directly execute our Python file in the directory **_(Code/close_models)_** to perform inferencing of the zero-shot, few-shot, provided that you prepare a key:
```
python gemini_2-5_zero_shot.py
python gemini_2-5_one_shot.py
python gpt4o_zero_shot.py
python gpt4o_one_shot.py
```
Gemini needs to apply on the [official website](https://aistudio.google.com/app/apikey), and GPT4 needs to be purchased on the [official website](https://openai.com/).

### Evaluation
You can process the results of model inference through the code we provide to calculate overall accuracy, overall P, R, F1 indicators, and use a graphical interface to statistically calculate the accuracy rate of each subtask. We integrate the calculation process into the Python files in the directory **_(Code/eval)_**:
```
python calculate_prf1.py
python calculate_acc.py
```

### Requirements
The environment configuration required for debugging code is placed in directory **_(Code/requirement)_**

## 4 License
This project is licensed under the [Apache-2.0 License](LICENSE).
