import math
import os
import json

filelist = {}


def build_index_from_files(given_directory):
    inverted_index = {}
    global filelist
    filelist = os.listdir(given_directory)
    for filename in filelist:
        filepath = os.path.join(given_directory, filename)
        doc_id = filename  # 使用文件名作为文档ID
        with open(filepath, 'r') as file:
            for line in file:
                stems = line.strip().split()  # 分割每行为词干列表
                for stem in stems:
                    if stem in inverted_index:
                        inverted_index[stem].add(doc_id)
                    else:
                        inverted_index[stem] = {doc_id}
    return inverted_index


print("Current Working Directory:", os.getcwd())
directory = "../4107_output/"
index = build_index_from_files(directory)
serializable_inverted_index = {word: list(doc_ids) for word, doc_ids in index.items()}
with open('inverted_index.json', 'w') as json_file:
    json.dump(serializable_inverted_index, json_file)

idf_table = {}
total_documents = len(filelist)  # 文档总数
for word, documents in index.items():
    idf_table[word] = math.log((total_documents / (len(documents)) )+ 0.01)  # avoid 0 idf
with open("idf_table.json", "w") as json_file:
    json.dump(idf_table, json_file)


def build_tf(given_directory, given_doc):
    tf_table = {}
    with open(os.path.join(given_directory, given_doc), "r") as file:
        for line in file:
            words = line.strip().split()
            for word in words:
                tf_table[word] = tf_table.get(word, 0) + 1
    return tf_table


tf_tables = {}
for doc_id in filelist:
    tf_tables[doc_id] = build_tf(directory, doc_id)

with open('tf_tables.json', 'w') as json_file:
    json.dump(tf_tables, json_file)
