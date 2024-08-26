
import pickle
import json

# 读取 .pkl 文件
with open('/mnt/disk_1/tengbo/peract_bimanual/instructions.pkl', 'rb') as file:
    data = pickle.load(file)



# 写入 .txt 文件
with open('output_file.txt', 'w', encoding='utf-8') as file:
    file.write(str(data))