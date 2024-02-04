import spacy
import os
import re
import requests
import time

# 加载英文模型
nlp = spacy.load('en_core_web_sm')

# 从提供的链接下载停用词列表
stop_words_url = "https://www.site.uottawa.ca/~diana/csi4107/StopWords"
stop_words = set(requests.get(stop_words_url).text.splitlines())


def preprocess_text_spacy(text):
    doc = nlp(text)
    filtered_tokens = [token.text.lower() for token in doc if token.text.lower() not in stop_words and token.is_alpha]
    return filtered_tokens


def process_documents(input_directory, output_directory):
    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    total_files = len(os.listdir(input_directory))
    processed_files = 0

    for filename in os.listdir(input_directory):
        start_time = time.time()

        input_filepath = os.path.join(input_directory, filename)
        output_filepath = os.path.join(output_directory, filename + '_processed.txt')

        with open(input_filepath, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()

            docs = re.findall(r'<DOC>.*?<\/DOC>', content, re.DOTALL)
            processed_docs = []
            for doc in docs:
                text_content = re.search(r'<TEXT>(.*?)<\/TEXT>', doc, re.DOTALL)
                if text_content:
                    text = text_content.group(1)
                    processed_text = preprocess_text_spacy(text)
                    processed_docs.append(' '.join(processed_text))

            with open(output_filepath, 'w', encoding='utf-8') as out_file:
                out_file.write('\n'.join(processed_docs))

        end_time = time.time()
        processed_files += 1
        print(f"Processed file {processed_files}/{total_files} ({filename}) in {end_time - start_time:.2f} seconds")


# 指定您的数据集目录和输出目录
dataset_directory = "/Users/frankxu/Documents/4107_data/"
output_directory = "/Users/frankxu/Documents/4107_temp/"
process_documents(dataset_directory, output_directory)
