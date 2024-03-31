import math
import os
import argparse
import json
import fire
from tqdm import tqdm
from data_processing.utils import load_jsonl, save_jsonl, load_json, save_json, set_seed
from multiprocessing.pool import Pool
import random
import openai
import numpy as np
import re
from logzero import logger

import base64
from io import BytesIO
from PIL import Image
from data_processing.utils import format_docstring

eval_prompt = '''Your task is to act as an impartial and objective assessor of answers generated by a Large Multimodal Model (LMM) for video-based questions. Utilizing video frames, a posed question, and the model's provided answer, your evaluation should focus on the following aspects:

- **Relevance**: Does the predicted answer directly address the question posed, considering the information provided in the video caption?
- **Accuracy**: Compare the predicted answer to the ground truth answer. Does the prediction accurately reflect the information given in the ground truth answer without introducing factual inaccuracies?
- **Clarity**: Assess the clarity of the predicted answer. Look for issues such as repetition, unclear descriptions, or any grammatical errors that could hinder understanding.
- **Completeness**: Determine if the predicted answer fully covers the scope of the ground truth answer. Does it leave out critical information or does it include all necessary details?

**Input**:
Question: {question}
Model Predicted Answer: {prediction}

**Output Format**:
Explanation: <brief judgement of prediction>
Score: <an integer score of quality from 1-5>
'''

client = openai.AzureOpenAI(
    azure_endpoint="https://search-va.byteintl.net/gpt/openapi/online/multimodal/crawl/",
    api_version="2023-09-01-preview",
    api_key="zo5Pr2NPcvmS6alcZpuAKD6UG5H0oIgl",
)

def parse_output(result):
    judgment_pattern = r"Judgment:\s*(.*?)\s*Score:"
    score_pattern =  r"Score:\s*(.*?)\s*Reference Caption:"
    referenc_pattern = r"Reference Caption:\s*(.*?)$"
    judgment = re.findall(judgment_pattern, result, re.MULTILINE | re.DOTALL)
    score = re.findall(score_pattern, result, re.MULTILINE | re.DOTALL)
    reference = re.findall(referenc_pattern, result, re.MULTILINE | re.DOTALL)

    return judgment[0], score[0], reference[0]

def gpt4vcaption(client, content):
    # try:
    completion = client.chat.completions.create(
        model="gptv",
        max_tokens = 500,
        messages=[{
                "role": "user",
                "content": content,
            }],
        timeout=120,
    )

    output = completion.choices[0].message.content
    # except:
    #     print('can not parse request...')
    #     output = None

    return output

def gpt4v_eval(client, frame_root, question, prediction):
    prompt = eval_prompt
    prompt = format_docstring(prompt)
    prompt = prompt.format(question=question, prediction=prediction)
    print(prompt)
    base64frames = []
    frame_names = os.listdir(frame_root)
    frame_names.sort()
    # print(f"frame names: {frame_names}")
    num_frames = 8
    if len(frame_names) <= num_frames:
        frame_id_list = list(range(len(frame_names)))
    else:
        duration = len(frame_names)
        frame_id_array = np.linspace(0, duration-1, num_frames, dtype=int)
        frame_id_list = frame_id_array.tolist()

    # print(f"selected frame names: {frame_id_list}")
    
    for frame_id in frame_id_list:
        frame_name = frame_names[frame_id]
        frame_path = os.path.join(frame_root, frame_name)
        new_image = Image.open(frame_path)
        buffered = BytesIO()
        new_image.save(buffered, format="JPEG")
        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
        base64frames.append(img_b64_str)
    
    content = [{"type": "text","text": prompt}]

    for base64image in base64frames:
        a_data = {
            "type": "image_url",
            "image_url": {
            "url": f"data:image/jpeg;base64,{base64image}"
            }
        }
        content.append(a_data)
    #output_text = gpt4vcaption(client, content)
    retry=0
    while retry < 70:
        try: 
            output_text = gpt4vcaption(client, content)
            if output_text:
                break
        except Exception as e:
            output_text = None
            retry += 1
            print(f"retry {retry}: {e}")

    # output = gpt4vcaption(client, content)

    return output_text

def annotate(data, output_dir):
    for i, item in tqdm(enumerate(data)):
        try:
            frame_root = item['video']
            question = item['prompt']
            prediction = item['prediction']
            result = gpt4v_eval(client=client, frame_root=frame_root, question=question, prediction=prediction)
            # result = debug_result
            result_dict = item.copy()
            result_dict['gpt4v_response'] = result
            save_json(f"{output_dir}/{item['id']}.json", result_dict)
        except Exception as e:
            print(f"Error processing file '{item['id']}': {e}")

def main(data_path, output_path, output_dir, num_tasks, **kwargs):
    data = load_jsonl(data_path)

    while True:
        try:
            if os.path.exists(output_path):
                combined_contents = load_jsonl(output_path)
                result_idx = [item['id'] for item in combined_contents]
                # missing_data = [item for i, item in enumerate(data) if i not in result_idx]
                missing_data = [item for item in data if item['id'] not in result_idx]
                data = missing_data
                logger.info(f"found resulting file, impute missing data: {len(data)}")
            else:
                combined_contents = []

            # Break the loop when there are no incomplete files
            if len(data) == 0:
                break
            if len(data) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            part_len = len(data) // num_tasks
            all_parts = [data[i:i + part_len] for i in range(0, len(data), part_len)]
            task_args = [(part, output_dir) for part in all_parts]

            # annotate(data, output_dir)
            # Use a pool of workers to process the files in parallel.
            with Pool() as pool:
                pool.starmap(annotate, task_args)

            print("finish iteration")
            # combine

            for file_name in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file_name)
                with open(file_path, "r") as json_file:
                    content = json.load(json_file)
                    combined_contents.append(content)
                os.remove(file_path)
            save_jsonl(output_path, combined_contents)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    fire.Fire(main)