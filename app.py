from flask import Flask
from flask import request, render_template, redirect, url_for

import re
import pickle
from keras.preprocessing.sequence import pad_sequences
from textblob import TextBlob
from transformer import *
from utils import *
from preprocessing import *


app = Flask(__name__)

def PreprocessSentence(line):

	line = line.encode('ascii', 'ignore').decode("utf-8")
	line = re.sub(r"[^\w+\s]+\s*\W*" , "" , line)
	line = line.lower()
	return line

def TokenizeSentence(text, eng_vocab):
	sen = []
	print("from tokenize ", eng_vocab.get("<unk>"))
	print("from tokenize ", eng_vocab["<unk>"])
	for i in text.split():
		try:
			sen.append(eng_vocab[i])
		except:
			sen.append(eng_vocab.get("<unk>"))
	return sen

def PadSentence(text, maxlen, padding = "post", value = 0):
	text = pad_sequences([text], maxlen= maxlen, padding= padding, value = value)
	return text

def transformer(text):
	global fren_vocab, eng_vocab, inverse_fren_vocab, inverse_eng_vocab, op_max_len
	global model, op_model
	
	sos_index = fren_vocab.get("<sos>")
	eos_index = fren_vocab.get("<eos>")
	print(text)
	text = decode_sentence(op_model, text ,fren_vocab , inverse_fren_vocab, op_max_len, sos_index, eos_index)
	return text

@app.route("/")
def hello():
	return render_template("index.html")

@app.route("/translate_page", methods = ["POST"])
def translate():

	text = str(request.form["input_text"])
	print(text)
	text = PreprocessSentence(text)
	print(text)
	text = TokenizeSentence(text, eng_vocab)
	print(text)
	text = PadSentence(text, maxlen = ip_max_len, padding = "post", value = 0)

	transformer_op = transformer(text)

	return "Transformer :" + transformer_op

if __name__ == "__main__":
	global eng_vocab, fren_vocab, inverse_eng_vocab, inverse_fren_vocab, ip_max_len, op_max_len, eng_vocab_len, fren_vocab_len,\
	d_model, d_inner, heads, dropout, d_key, layers, lr
	global model, op_model

	params_dir = "pickle/params.pkl"
	model_dir = "model/model_v1.h5"

	print("------------------loading params---------------")
	eng_vocab, fren_vocab, inverse_eng_vocab, inverse_fren_vocab, ip_max_len, op_max_len, eng_vocab_len, fren_vocab_len,\
	d_model, d_inner, heads, dropout, d_key, layers, lr = pickle.load(open(params_dir, "rb"))
	print(ip_max_len, op_max_len)

	print("--------------------loading models-----------------")
	m = transfomer(ip_max_len = ip_max_len, op_max_len = op_max_len, 
	d_model = d_model, d_inner = d_inner, n_head = heads, dropout =dropout, d_q = d_key, d_k=d_key, d_v = d_key, n_layers = layers, 
	eng_vocab_len = eng_vocab_len , fren_vocab_size = fren_vocab_len, lr = lr)
	model , op_model = m.create_model()
	op_model.load_weights(model_dir)
	op_model._make_predict_function()

	app.run(debug = True)


























