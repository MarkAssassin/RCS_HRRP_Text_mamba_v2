import scipy.io
import numpy as np
from openai import OpenAI
import time
from tqdm import tqdm
import json
import os
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt

# 该文件用于调用大模型的API生成HRRP的文本描述

def load_data(freq, snr):
    # rcs
    # data_path_train = f'text_creator_v2/data/rcs_train_data_{freq}_{snr}dB.mat'
    # rcs_data_train = scipy.io.loadmat(data_path_train)
    # data_path_test = f'text_creator_v2/data/rcs_test_data_{freq}_{snr}dB.mat'
    # rcs_data_test = scipy.io.loadmat(data_path_test)
    # rcs_data_train = abs(rcs_data_train['train_data'])
    # rcs_data_test = abs(rcs_data_test['test_data'])
    # hrrp
    data_path_train = f'text_creator_v2/data/hrrp_train_data_{freq}_{snr}dB.mat'
    hrrp_data_train = scipy.io.loadmat(data_path_train)
    data_path_test = f'text_creator_v2/data/hrrp_test_data_{freq}_{snr}dB.mat'
    hrrp_data_test = scipy.io.loadmat(data_path_test)
    hrrp_data_train = abs(hrrp_data_train['train_data']) # (101,101,4840)
    hrrp_data_test = abs(hrrp_data_test['test_data']) # (101,101,1452)
    # data_rcs = np.concatenate((rcs_data_train, rcs_data_test), axis=0)
    data_hrrp = np.concatenate((hrrp_data_train, hrrp_data_test), axis=2)
    # 归一化
    data_hrrp = (data_hrrp-data_hrrp.min())/(data_hrrp.max()-data_hrrp.min()) * 255
    # plt.imshow(data_hrrp[:,:,0])
    # plt.show()
    return data_hrrp

def round_array(arr, decimals=6):
    if arr.ndim == 1:
        return [round(float(x), decimals) for x in arr]
    elif arr.ndim == 2:
        return [[round(float(x), decimals) for x in row] for row in arr]
    else:
        raise ValueError("Only support 1D or 2D array")

def get_imgs_base64(hrrp_image):

    hrrp_image = Image.fromarray(hrrp_image)
    isar_img = hrrp_image.convert('L')
    # render_img = Image.open(imgs_path['render']).convert('L')

    isar_buf = io.BytesIO()
    # render_buf = io.BytesIO()

    # 保存到内存
    isar_img.save(isar_buf, format='PNG', optimize=True)

    # render_img.thumbnail((256, 256), Image.BILINEAR)
    # render_img.save(render_buf, format='PNG', optimize=True)

    # 转base64
    image_b64_list = {
        "hrrp": base64.b64encode(isar_buf.getvalue()).decode("utf-8"),
        # "render": base64.b64encode(render_buf.getvalue()).decode("utf-8"),
    }

    # print(f"isar {isar_img.size} : {isar_buf.tell()/1024:.2f} KB")
    # print(f"render {render_img.size} : {render_buf.tell()/1024:.2f} KB")

    return image_b64_list

if __name__ == '__main__':
    # 初始化客户端
    client = OpenAI(
    api_key="sk-1e0845e4621f4a8da8504b5916f6ecc5",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" # 华北2(北京)
    )
    
    # 设置prompt
    # prompt = '''
    #     You are an expert in radar micro-motion target recognition. Please provide a professional textual description of the given HRRP image, without including any specific numbers. The description should cover, but is not limited to:
    #     1. HRRP image: the given image is HRRP image, scattering center distribution, echo intensity patterns, temporal modulation characteristics, stability and regularity of scattering centers, presence of multiple scattering centers and their interactions.
    #     2. Overall signal characteristics: signal stability, combination of regularity and randomness, reflection of micro-motion information, prominent patterns or anomalies.
    #     3. Text style and length requirements: the description should be concise, professional, and highlight micro-motion target characteristics. The output must be between 100 and 120 words, suitable for training text encoders or feature alignment, and must not include any numerical values.
    #     Focus on describing the shape, trend, and pattern of HRRP features, providing a clear, comprehensive summary of micro-motion characteristics.
    #     '''

    # 目标1和3的Prompt(2根线)
    prompt_13 = '''
        You are an expert in radar micro-motion target recognition. Please provide a professional textual description of the given HRRP image, focusing on two bright scattering lines (strong echo bands).

        The output MUST follow this fixed structure:

        - First sentence: describe that there are two bright scattering bands, and explicitly describe their relative spatial distance and vertical arrangement (e.g., close or far apart, upper/lower relationship).
        - Second sentence: describe the intensity characteristics, stability, and structural appearance of each scattering band separately (upper band and lower band).
        - Third sentence: describe the temporal evolution pattern of both bands, emphasizing their modulation behavior such as periodicity, synchronization, similarity, or differences.

        Requirements:
        - The output must be concise and professional.
        - STRICTLY follow the 3-sentence structure above.
        - Do NOT include any numerical values.
        - Focus on spatial relationship, band-specific characteristics, and temporal variation trends.
        - Avoid physical derivations; emphasize visual and pattern-level description only.
        - The total length must be strictly between 50 and 60 words.
    '''

    prompt_24 = '''
        You are an expert in radar micro-motion target recognition. Please provide a professional textual description of the given HRRP image, focusing on three bright scattering lines (strong echo bands).

        The output MUST follow this fixed structure:

        - First sentence: describe that there are three bright scattering bands, and explicitly describe their spatial arrangement and relative distances (e.g., upper, middle, lower; close or well-separated relationships).
        - Second sentence: describe the intensity characteristics, stability, and visual appearance of each scattering band separately (upper, middle, and lower bands).
        - Third sentence: describe the temporal evolution pattern of all bands, focusing on modulation behavior such as periodicity, synchronization, continuity, and possible differences among bands.

        Requirements:
        - The output must be concise and professional.
        - STRICTLY follow the 3-sentence structure above.
        - Do NOT include any numerical values.
        - Emphasize spatial relationships, band-wise characteristics, and temporal variation trends.
        - Avoid physical derivations; focus only on visual and pattern-level description.
        - The total length must be strictly between 50 and 60 words.
    '''

    # 下载数据
    freq = 'L' # L/S/X
    snr = 20
    data_hrrp = load_data(freq, snr) # (6292, 101) and (101, 101, 6292)   
    # 调用 Qwen
    print("<------ Starting to create text data using Qwen API ------>")
    text_data_list = []
    for i in tqdm(range(data_hrrp.shape[2])):
    # idx_list = [0, 1500, 3000, 4800]
    # for i in idx_list:
        # 根据样本索引选择对应的HRRP，并生成文本描述
        if 0 <= i <= 1209 or 4840 <= i <= 5202:
            prompt = prompt_13
        elif 1210 <= i <= 2419 or 5203 <= i <= 5565:
            prompt = prompt_24
        elif 2420 <= i <= 3629 or 5566 <= i <= 5928:
            prompt = prompt_13
        else:
            prompt = prompt_24
        hrrp_image = data_hrrp[:,:,i]
        hrrp_image = round_array(hrrp_image, 6)
        plt.imshow(data_hrrp[:,:,i])
        plt.show()
        hrrp_base64 = get_imgs_base64(hrrp_image) # 将HRRP图像转换为base64格式，减少prompt长度

        # 将RCS序列和HRRP图像的特征信息提取出来，构造一个新的prompt
        response = client.chat.completions.create(
            model="qwen-flash",   # qwen-plus / qwen-max / qwen-flash 模型列表: https://help.aliyun.com/model-studio/getting-started/models
            messages=[
                {"role": "system", "content": "You are an expert in radar micro-motion target recognition."},
                {"role": "user", 
                #  "content": prompt_plus,
                 "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/png;base64,{hrrp_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
                }
            ],
            temperature=0.7
        )
        # print(f"Generated text for sample {i}: {response.choices[0].message.content}")
        text_data_list.append(response.choices[0].message.content)
    file_path = f'text_creator_v2/data/data_{freq}_{snr}dB_api.npy'
    np.save(file_path, np.array(text_data_list))
    print("<------ Text data creation completed ------>")

    
    # 将message存成jsonl格式,传给LZS使用A100调用API进行文本生成
    # print("<------ Starting to create jsonl data for API ------>")
    # jsonl_path = f'text_creator_v2/data/data_{freq}_{snr}dB_api.jsonl'
    # os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    # with open(jsonl_path, "w", encoding="utf-8") as f:
    #     for i in tqdm(range(data_rcs.shape[0])):
    #         rcs_sequence = data_rcs[i,:]
    #         hrrp_image = data_hrrp[:,:,i]
    #         rcs_list = round_array(rcs_sequence, 6)
    #         hrrp_list = round_array(hrrp_image, 6)
    #         # 将RCS序列和HRRP图像的特征信息提取出来，构造一个新的prompt
    #         prompt_plus = prompt + f"\n\nRCS sequence: {rcs_list}\nHRRP image: {hrrp_list}" # prompt的token太长，后续建议修改(如将HRRP图像转为base-64格式)
    #         messages=[
    #                 # {"role": "system", "content": "You are an expert in radar micro-motion target recognition."},
    #                 {"role": "user", "content": prompt_plus}
    #             ]
    #         sample = {
    #             "id": i,
    #             "messages": messages
    #         }
    #         # 写入一行
    #         f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    # print("<------ JSONL data creation completed ------>")

