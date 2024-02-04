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

query_doc = "topics1-50.txt"  # the path of query document


def preprocess_text_spacy(text):
    doc = nlp(text)
    filtered_tokens = [token.text.lower() for token in doc if token.text.lower() not in stop_words and token.is_alpha]
    return filtered_tokens


def preprocess_text_query(text):
    with open(query_doc, "r") as f:
        doc = f.read()
