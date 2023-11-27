from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from PIL import Image
import gradio as gr
import os
import torch
torch.manual_seed(1234)


model_name = ["Qwen-VL-Chat", 
              "Qwen-VL-Chat_ShareGPT4v-23K", 
              "Qwen-VL-Chat_ShareGPT4v-50K"
              ]
model_dirs = ["Qwen/Qwen-VL-Chat", 
              "/mnt/petrelfs/lijingsong/MLLM/ShareGPT4V/Experiments/Qwen-VL/output_qwen-chat_23k_ours", 
              "/mnt/petrelfs/lijingsong/MLLM/ShareGPT4V/Experiments/Qwen-VL/output_qwen-chat_50k_ours"
              ]

tokenizer1 = AutoTokenizer.from_pretrained(model_dirs[0], trust_remote_code=True)
model1 = AutoModelForCausalLM.from_pretrained(model_dirs[0], device_map="cuda", trust_remote_code=True).eval()
model1.generation_config = GenerationConfig.from_pretrained(model_dirs[0], trust_remote_code=True)

tokenizer2 = AutoTokenizer.from_pretrained(model_dirs[1], trust_remote_code=True)
model2 = AutoModelForCausalLM.from_pretrained(model_dirs[1], device_map="cuda", trust_remote_code=True).eval()
model2.generation_config = GenerationConfig.from_pretrained(model_dirs[1], trust_remote_code=True)

tokenizer3 = AutoTokenizer.from_pretrained(model_dirs[2], trust_remote_code=True)
model3 = AutoModelForCausalLM.from_pretrained(model_dirs[2], device_map="cuda", trust_remote_code=True).eval()
model3.generation_config = GenerationConfig.from_pretrained(model_dirs[2], trust_remote_code=True)


def detailed_caption(img_path, prompt):
    query1 = tokenizer1.from_list_format([
            {'image': img_path},
            {'text': prompt},
        ])
    response1, history1 = model1.chat(tokenizer1, query=query1, history=None)

    query2 = tokenizer2.from_list_format([
            {'image': img_path},
            {'text': prompt},
        ])
    response2, history2 = model2.chat(tokenizer2, query=query2, history=None)

    query3 = tokenizer3.from_list_format([
            {'image': img_path},
            {'text': prompt},
        ])
    response3, history3 = model3.chat(tokenizer3, query=query3, history=None)

    return response1, response2, response3



gr.close_all()
demo = gr.Interface(fn=detailed_caption,
                    inputs=[gr.Image(label="Image", type="filepath"), gr.Textbox(label="Prompt", value="Describe this image in detail.")],
                    outputs=[gr.Textbox(label=model_name[0]),
                             gr.Textbox(label=model_name[1]),
                             gr.Textbox(label=model_name[2])],
                    title="Qwen-VL-Chat",
                    allow_flagging="never"
                    )
demo.launch(share=True, server_name='0.0.0.0', server_port=10001)