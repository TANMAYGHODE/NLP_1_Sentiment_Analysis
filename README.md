# NLP_1_Sentiment_Analysis
This is a simple project for movie review wiht BERT embeddings. 
Inference is by using FASTAPI for any movie review.
frontend UI is by using a React.js framework.

# How to run :

1. Pull nvidia-docker for gpu dependency.
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

2. Pull Tensorflow docker for tensorflow training.
docker run --gpus all -it  -v $(pwd):/workspace tensorflow/tensorflow:latest-gpu /bin/bash 
for docker tensorflow image. 

In docker CLI install the following dependencies:

pip install "tensorflow-text==2.8.*"
pip install tf-models-official==2.7.0
pip install tfds-nightly

