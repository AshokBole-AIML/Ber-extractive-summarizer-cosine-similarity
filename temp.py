import os,os.path
my_path = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(*[my_path, "summarizer\distilroberta-base-paraphrase-v1"])