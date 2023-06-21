from pathlib import Path
import json

from collections import Counter
import numpy as np
import mygrad as mg

# load COCO metadata
'''filename = r"C:/Users/HyoJP/Desktop/BWSI/week3/capstone/data/captions_train2014.json"
with Path(filename).open() as f:
    coco_data = json.load(f)

from gensim.models import KeyedVectors
filename = r"C:/Users/HyoJP/Desktop/BWSI/week3/capstone/data/glove.6B.200d.txt.w2v"
glove = KeyedVectors.load_word2vec_format(filename, binary=False)'''

import re, string

punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))

def strip_punc(corpus):
    return punc_regex.sub('', corpus)

def create_embeddings(coco_data, glove):
    embeddings = {}
    word_count = Counter()

    for caption_info in coco_data["annotations"]:
        caption = caption_info["caption"]
        caption = strip_punc(caption).lower().split()
        word_count.update(Counter(set(caption)))

    for caption_info in coco_data["annotations"]:
        caption = caption_info["caption"]
        caption_id = caption_info["id"]
        caption = strip_punc(caption).lower().split()
        w_caption = np.zeros(200)
        for word in caption:
            idf = np.log10(len(coco_data["annotations"]) / word_count[word])
            if word not in glove:
                w_word = np.zeros(200)
            else:
                w_word = glove[word]
            w_caption += idf * w_word
        w_caption = w_caption / mg.sqrt((w_caption ** 2).sum(keepdims=True))
        embeddings[caption_id] = w_caption.data
    return embeddings

def create_embedding(query, coco_data, glove):
    
    query = strip_punc(query).lower().split()
    word_count = Counter()

    for caption_info in coco_data["annotations"]:
        caption = caption_info["caption"]
        caption = strip_punc(caption).lower().split()
        word_count.update(Counter(set(caption)))
        
    w_query = np.zeros(200)
    for word in query:
        if word in word_count:
            idf = np.log10(len(coco_data["annotations"]) / word_count[word])
        else:
            idf = 0

        if word not in glove:
            w_word = np.zeros(200)
        else:
            w_word = glove[word]
            
        w_query += idf * w_word
    w_query = w_query / mg.sqrt((w_query ** 2).sum(keepdims=True))
    embedding = w_query.data
    return w_query