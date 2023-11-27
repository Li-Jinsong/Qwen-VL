import argparse
import torch
from tqdm import tqdm
import json

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from PIL import Image

import requests
from PIL import Image
from io import BytesIO


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()

    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cuda', trust_remote_code=True).eval()
    model_path = args.model_path
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path, # path to the output directory
        device_map="cuda",
        trust_remote_code=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    
    with open(args.questions_file) as f:
        llvqa_data = json.load(f)  
        
    for i, llddata in enumerate(tqdm(llvqa_data)):
        filename = llddata["img_path"]
        if args.lang == "en":
            message = llddata["question"] + "\nChoose between one of the options as follows:\n"
        elif args.lang == "zh":
            message = llddata["question"] + "\在下列选项中选择一个:\n"
        else:
            raise NotImplementedError("Q-Bench does not support languages other than English (en) and Chinese (zh) yet. Contact us (https://github.com/VQAssessment/Q-Bench/) to convert  Q-Bench into more languages.")
        for choice, ans in zip(["A.", "B.", "C.", "D."], llddata["candidates"]):
            message += f"{choice} {ans}\n"
        qs = message
        
        img_dir = '/mnt/petrelfs/lijingsong/MLLM/Benchmark/qbench/images_llvisionqa/'
        query = f'Picture 1: <img>{img_dir + filename}</img>\n{qs}'
        with torch.inference_mode():
            outputs, _ = model.chat(tokenizer, query=query, history=None)

        outputs = outputs.strip()
        llddata["response"] = outputs
        with open(args.answers_file, "a") as wf:
            json.dump(llddata, wf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="llava-v1.5")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="./playground/data/qbench/images_llvisionqa")
    parser.add_argument("--questions-file", type=str, default="./playground/data/qbench/llvisionqa_dev.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--lang", type=str, default="en")
    args = parser.parse_args()

    eval_model(args)