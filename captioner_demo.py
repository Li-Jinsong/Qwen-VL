from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from PIL import Image
import gradio as gr
import os
import torch
torch.manual_seed(1234)


model_name = "share-captioner"
model_dirs = "/mnt/petrelfs/share_data/lijinsong/Models/qwen-100k"

tokenizer = AutoTokenizer.from_pretrained(model_dirs, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dirs, device_map="cuda", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained(model_dirs, trust_remote_code=True)

def detailed_caption(img_path, prompt):
    prompt = 'Describe this image in detail.'
    query = tokenizer.from_list_format([
            {'image': img_path},
            {'text': prompt},
        ])
    response, history = model.chat(tokenizer, query=query, history=None)
    return response

with gr.Blocks() as demo:
    gr.Markdown("# Share-Captioner")
    with gr.Row():
        with gr.Column():
            img_path = gr.Image(label="Image", type="filepath")
            cap_btn = gr.Button("Start")
        with gr.Column():
            caption = gr.Textbox(label='Caption')

    cap_btn.click(detailed_caption, inputs = [img_path], outputs = [caption])

demo.launch(share=True, server_name='0.0.0.0', server_port=10011)