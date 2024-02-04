import json

# 假设TF表存储在'./tf_table.json'文件中
file_path = './tf_tables.json'
file_path1 = './idf_table.json'
output_name = 'tf_idf_table.json'
output_dict = {}

# 从文件中读取数据
with open(file_path, 'r') as file:
    tf_table_dict = json.load(file)

for key in tf_table_dict:
    print(key)
    tf_values = tf_table_dict[key]
    # IDF字典
    # 从JSON文件读取数据
    with open(file_path1, 'r', encoding='utf-8') as file1:
        idf_table_dict = json.load(file1)

    # 计算TF-IDF值
    tf_idf_values = {word: tf_values[word] * idf_table_dict.get(word, 0) for word in tf_values}
    output_dict[key] = tf_idf_values


# 将字典写入JSON文件
with open(output_name, 'w', encoding='utf-8') as f:
    json.dump(output_dict, f, ensure_ascii=False, indent=4)