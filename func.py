'''
这个文件搞一些函数
'''
from config import *
import json
import os


def get_w2i_dict():
    # 读取文字转ID的字典并生成逆运算字典，即ID转文字
    if os.path.exists(word2idx_path):
        with open(word2idx_path, 'r+') as f:
            w2i = json.loads(f.read())
        i2w = {}
        for i in w2i:
            i2w[w2i[i]] = i
        return w2i, i2w
    return None, None
