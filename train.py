import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import tensorflow_text as text  # A dependency of the preprocessing model
import tensorflow_addons as tfa
from official.nlp import optimization
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"


def get_sentence_embeding(sentences):
    preprocessed_text = bert_preprocess_model(sentences)
    return bert_model(preprocessed_text)['pooled_output']



bert_preprocess_model = hub.KerasLayer(preprocess_url)
bert_model = hub.KerasLayer(encoder_url)
df_1=pd.read_csv('IMDB_Dataset.csv')

review_train=[]

for i in df_1['review']:
 review_train.append(i)

for i, element in enumerate(review_train):
    element = element.replace('<br />','')
    element = element.replace('\' ','')
    review_train[i] = element

num_classes = 2
y_train=[]

for i in df_1['sentiment']:
    if i== 'positive':
        y_train.append(1)
    else:
        y_train.append(0)

y_train = keras.utils.to_categorical(y_train, num_classes)


preprocessed_text=[]

count=0

while count < len(y_train):
  pre=get_sentence_embeding(review_train[count:count+500])
  preprocessed_text.append(pre)
  count=count+500

a=preprocessed_text[0]
for i in range(1, len(preprocessed_text)):
  a=np.append(a,preprocessed_text[i],axis=0)

input_shape = a.shape[1]
Xvec = np.expand_dims(a, -1)
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Dense(10, activation="relu"),
        layers.Dense(10, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

batch_size = 2
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(Xvec, y_train, batch_size=batch_size, epochs=epochs)
model.save('my_model.h5')
