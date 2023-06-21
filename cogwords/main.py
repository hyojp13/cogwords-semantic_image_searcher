import pickle
import numpy as np
import mygrad
from PIL import Image
from embedding import create_embedding
import os
from database import database
from embedding import create_embeddings
import pickle
import shutil


db = database("/Users/KittyKat/BWSI/BWSI-program/Week-3/captions_train2014.json")


db.load_data("./img_weights.pkl", './glove.pkl')

def find_images(caption, k):
    embedding = create_embedding(caption, db.coco_data, db.glove)
    IDs = db.query(embedding, k=k)
    os.mkdir("./pics")
    for id in IDs:
        img = db.display_image(id)
        img.save(f"./pics/{str(id)}.png")
    return IDs

def delete_imgs():
    shutil.rmtree("./pics")