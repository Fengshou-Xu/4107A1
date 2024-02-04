import json
import os
import re

import requests
import spacy

# print("Current Working Directory:", os.getcwd())
# given_directory = "../4107_output"
# result = os.listdir(given_directory)
# print(result)


doc = """
<top>

<num>1


<title>Coping with overcrowded prisons

<desc>

The document will provide information on jail and prison overcrowding and 
how inmates are forced to cope with those conditions; or it will reveal 
plans to relieve the overcrowded condition.

<narr>

A relevant document will describe scenes of overcrowding that have 
become all too common in jails and prisons around the country.  The 
document will identify how inmates are forced to cope with those 
overcrowded conditions, and/or what the Correctional System is doing, 
or planning to do, to alleviate the crowded condition.

</top>
"""

query = re.findall(r'<top>(.*?)</top>', doc, re.DOTALL)
print(query)
num = re.search(r'<num>(.*?)\n', query[0])
doc_num = 0
if num:
    doc_num = num.group(1).strip()
    print(doc_num)

doc_content = re.search(r'<title>([\s\S]*)', query[0])
doc_title_text = ""
if doc_content:
    doc_title_text = doc_content.group(1).strip()
    print(doc_title_text)

remove_element_text = re.sub(r'</?desc>|</?narr>', '', doc_title_text)
print("cleaned text : \n")
print(remove_element_text)

nlp = spacy.load('en_core_web_sm')

# 从提供的链接下载停用词列表
stop_words_url = "https://www.site.uottawa.ca/~diana/csi4107/StopWords"
stop_words = set(requests.get(stop_words_url).text.splitlines())


def preprocess_text_spacy(text):
    doc = nlp(text)
    filtered_tokens = [token.text.lower() for token in doc if token.text.lower() not in stop_words and token.is_alpha]
    return filtered_tokens


processed_text = preprocess_text_spacy(remove_element_text)
print(processed_text)

query_tf_table = {}
for word in processed_text:
    query_tf_table[word] = query_tf_table.get(word, 0) + 1

with open('idf_table.json', 'r') as idf_table_file:
    idf_table = json.load(idf_table_file)

query_tfidf_vector = {}
for word in processed_text:
    word_tf = query_tf_table.get(word)
    word_idf = idf_table.get(word,0)
    print(word, word_tf, word_idf)
    query_tfidf_vector[word] = word_tf * word_idf

print(query_tfidf_vector)


