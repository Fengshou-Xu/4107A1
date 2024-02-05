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
    max_query_freq = max(query_tf_table.values())
    print(word, word_tf, word_idf, max_query_freq)
    query_tfidf_vector[word] = (word_tf/max_query_freq) * word_idf

print(query_tfidf_vector)

with open('inverted_index.json', 'r') as idf_vector_file:
    inverted_index = json.load(idf_vector_file)

relate_doc = set()
for word in processed_text:
    if word in inverted_index:
        relate_doc.update(inverted_index[word])

print(relate_doc, "\n" , len(relate_doc))

with open('tf_idf_table.json' , 'r') as tfidf_table_file:
    tf_idf_table = json.load(tfidf_table_file)

relate_doc_tfidf = {}
for doc in relate_doc:
    relate_doc_tfidf[doc] = tf_idf_table.get(doc)

dot_product = 0
query_norm = 0
doc_norm = 0
cos_similarities = {}

for doc in relate_doc:
    with open('../4107_output/'+doc , 'r') as this_doc:
        this_doc_content = this_doc.read()
        this_doc_tfidf = relate_doc_tfidf[doc]
        for word in processed_text:
            if word in this_doc_content:
                query_word_tfidf = query_tfidf_vector.get(word, 0)
                doc_word_tfidf = this_doc_tfidf.get(word, 0)
                dot_product += query_word_tfidf * doc_word_tfidf
    for word in query_tfidf_vector:
        query_norm += query_tfidf_vector.get(word) ** 2

    for word in this_doc_tfidf:
        doc_norm += this_doc_tfidf.get(word) ** 2

    query_norm = query_norm ** 0.5
    doc_norm = doc_norm ** 0.5

    if query_norm == 0 or doc_norm == 0:
        similarity = 0
    else:
        similarity = dot_product / (query_norm * doc_norm)
    cos_similarities[doc] = similarity

sorted_cos_similarities = sorted(cos_similarities.items(), key=lambda item: item[1], reverse=True)

with open('cos_similarities.json', 'w') as cos_similarities_file:
    for doc, value in sorted_cos_similarities:
        cos_similarities_file.write( str(doc) + " : " + str(value) + "\n")





