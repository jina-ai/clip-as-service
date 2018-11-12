# bert-as-service

Using BERT model as a sentence encoding service.

## What is it?

[Developed by Google](https://github.com/google-research/bert), BERT is a method of pre-training language representations. It leverages an enormous amount of plain text data publicly available on the web and is trained in an unsupervised manner. Pre-training a BERT model is fairly expensive but one-time procedure for each language. Google released several pre-trained models where [you can download from here](https://github.com/google-research/bert#pre-trained-models).


## Usage

#### 1. Download the Pre-trained BERT Model
Download it from [here](https://github.com/google-research/bert#pre-trained-models), then uncompress the zip file into some folder say `/tmp/english_L-12_H-768_A-12/`


#### 2. Start a BERT service
```bash
python app.py -num_worker=4 -model_dir /tmp/english_L-12_H-768_A-12/
```
This will start a BERT service for four workers, meaning that it can handel up to four concurrent requests at the same time. (These workers are behind a simple load balancer.)

#### 3. Use Client to Encode Sentences
Now you can use pretrained BERT to encode sentences in your Python code simply as follows:
```python
ec = BertClient()
ec.encode(['abc', 'defg', 'uwxyz'])
```



