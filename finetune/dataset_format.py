import json
import os
import re


def replace_substring(conversations):
    pattern = re.compile(r'\[([0-9\.]*), ([0-9\.]*), ([0-9\.]*), ([0-9\.]*)\]')

    def repl(match):
        A = int(float(match.group(1)) * 1000)
        B = int(float(match.group(2)) * 1000)
        C = int(float(match.group(3)) * 1000)
        D = int(float(match.group(4)) * 1000)
        return f'<box>({A},{B}),({C},{D})</box>'
    
    for i in range(len(conversations)):
        try:
            conversations[i]['value'] = pattern.sub(repl, conversations[i]['value'])
        except Exception as e:
            print(e)
            print(conversations[i])
    return conversations


# TODO 1
sharegpt4v_path = '/mnt/petrelfs/lijingsong/MLLM/Datasets/sharegpt4v/llava_instruct_158k_share-cap-23k.json'
with open(sharegpt4v_path, 'r') as in_file:
    sharegpt4v = json.load(in_file)

qwen_vl = []
# TODO 2
img_dir = '/mnt/petrelfs/lijingsong/MLLM/Datasets/sharegpt4v/data'
for data in sharegpt4v:
    data_id = data['id']

    qwen_vl_conv = {
        "id": data_id,
        "conversations": []
    }
    qwen_vl_conv['conversations'] = data['conversations']
    flag= True
    for cnt in range(len(qwen_vl_conv['conversations'])):
        if '<img>' in qwen_vl_conv['conversations'][cnt]['value'] and '</img>' not in qwen_vl_conv['conversations'][cnt]['value']:
            flag = False  # 丢弃该数据
            break
        if cnt % 2 == 0:
            qwen_vl_conv['conversations'][cnt]['from'] = 'user'
            if '<image>' in qwen_vl_conv['conversations'][cnt]['value'] and 'image' in data.keys():
                image_path = os.path.join(img_dir, data['image'])
                qwen_image = 'Picture 1: <img>' + image_path + '</img>'
                qwen_vl_conv['conversations'][cnt]['value'] = qwen_vl_conv['conversations'][cnt]['value'].replace('<image>', qwen_image)
        else:
            qwen_vl_conv['conversations'][cnt]['from'] = 'assistant'
            if '<img>' in qwen_vl_conv['conversations'][cnt]['value']:
                flag= False  # 丢弃该数据
                break
    if 'image' in data.keys():
        qwen_vl_conv['conversations'] = replace_substring(qwen_vl_conv['conversations'])
    if flag:
        qwen_vl.append(qwen_vl_conv)

# TODO 3
qwen_vl_path = sharegpt4v_path[0:-5] + '-qwen.json'
with open(qwen_vl_path, 'w') as out_file:
    json.dump(qwen_vl, out_file, indent=4)