import streamlit as st
from keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle
import numpy as np
from PIL import Image
import os

# Paths
BASE_DIR = r'C:\Users\HP\projects\Final Project'
WORKING_DIR = r'C:\Users\HP\projects\Final Project\Working'
IMAGES_DIR = os.path.join(BASE_DIR, 'Images')
MODEL_PATH = r'C:\Users\HP\projects\Final Project\Working\best_model.h5'  # Updated path

with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()

# create mapping of image to captions
mapping = {}
# process lines
for line in (captions_doc.split('\n')):
    # split the line by comma(,)
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    # remove extension from image ID
    image_id = image_id.split('.')[0]
    # convert caption list to string
    caption = " ".join(caption)
    # create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    # store the caption
    mapping[image_id].append(caption)


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)  # Remove email addresses
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in words if word not in stop_words and len(word) > 2)  # Remove stopwords
    return text



def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            # Apply the preprocessing function
            caption = preprocess_text(caption)
            # Add start and end tags
            caption = 'startseq ' + caption + ' endseq'
            captions[i] = caption


all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

# Correct the path for the tokenizer
# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

# store features in pickle
pickle.dump(tokenizer, open(os.path.join(WORKING_DIR, 'tokenizer.pkl'), 'wb'))

with open(os.path.join(WORKING_DIR, 'tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)
# Load necessary data and models
@st.cache_resource
def load_resources():
    with open(os.path.join(WORKING_DIR, 'tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)


    # Load trained model
    model = load_model(MODEL_PATH)

    # Load VGG16 model
    vgg_model = VGG16()
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

    return model, tokenizer, vgg_model

# create data generator to get data in batch (avoids session crash)
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    # loop over images
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            # process each caption
            for caption in captions:
                # encode the sequence
                seq = tokenizer.texts_to_sequences([caption])[0]
                # split the sequence into X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pairs
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], 	num_classes=vocab_size)[0]
                    # store the sequences
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield {"image": X1, "text": X2}, y
                X1, X2, y = list(), list(), list()
                n = 0

# Predict caption for an image
def predict_caption(model, image, tokenizer, max_length=23):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Map index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Load resources
model, tokenizer, vgg_model = load_resources()
max_length = 35  # Replace with the actual max length from your training

# Streamlit app
st.title("Image Captioning App")

# Image upload
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    # Extract features using VGG16
    feature = vgg_model.predict(image, verbose=0)

    # Generate caption
    caption = predict_caption(model, feature, tokenizer, max_length)

    # Display the caption
    st.write("**Generated Caption:**")
    st.write(caption)
