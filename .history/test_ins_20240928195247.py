import pickle

# 替换为你的文件路径
file_path = '/mnt/disk_1/tengbo/peract_bimanual-real/18_lang_template.pkl'

# 打开并加载 .pkl 文件
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 打印内容
print(data)