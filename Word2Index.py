'''
这个文件是建立一个文字转ID的字典
'''
from tqdm import trange
from config import *
import json
import os

dir_name = 'HWDB1trn'
file_list = os.listdir(dir_name)

W2I = {}
has = 0

for i in trange(len(file_list)):
    file_name = os.path.join(dir_name, file_list[i])
    with open(file_name, 'rb+') as f:
        while f.readable():
            struct_size = int.from_bytes(f.read(4), byteorder='little')
            if struct_size == 0:
                break
            # 一个样本(包括该字段+label+宽+高+data字段)的字节数
            label = f.read(2).decode('gbk')  # 这个样本是什么字
            if label not in W2I:
                W2I[label] = has
                has += 1

            img = f.read(struct_size - 6)

print(len(W2I))
json_str = json.dumps(W2I)
with open(word2idx_path, 'w+') as f:
    f.write(json_str)
