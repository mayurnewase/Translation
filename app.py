from flask import Flask
from flask import request, render_template, redirect, url_for

import re
import pickle
from keras.preprocessing.sequence import pad_sequences
from textblob import TextBlob
from transformer import *
from utils import *
from preprocessing import *
from attention_model import *
import os

app = Flask(__name__)

def PreprocessSentence(line):

	line = line.encode('ascii', 'ignore').decode("utf-8")
	line = re.sub(r"[^\w+\s]+\s*\W*" , "" , line)
	line = line.lower()
	return line

def TokenizeSentence(text, eng_vocab):
	sen = []
	for i in text.split():
		try:
			sen.append(eng_vocab[i])
		except:
			sen.append(eng_vocab.get("<unk>"))
	print("after tokenizing {0}".format(sen))
	return sen

def PadSentence(text, maxlen, padding = "post", value = 0):
	text = pad_sequences([text], maxlen= maxlen, padding= padding, value = value)
	print("after padding {0}".format(text))
	return text

def transformer(text):
	global fren_vocab, eng_vocab, inverse_fren_vocab, inverse_eng_vocab, op_max_len
	global transformer_model
	
	sos_index = fren_vocab.get("<sos>")
	eos_index = fren_vocab.get("<eos>")
	text = decode_sentence_transformer(transformer_model, text ,fren_vocab , inverse_fren_vocab, op_max_len, sos_index, eos_index)
	return text

def attention(text):
	#decide by attention model
	global attention_model
	global att_eng_vocab, att_fren_vocab, att_ip_max_len, att_op_max_len, att_params, att_inv_eng_vocab, att_inv_fren_vocab
	sos_index = att_fren_vocab.get("<sos>")
	eos_index = att_fren_vocab.get("<eos>")

	text = decode_sentence_attention(attention_model, text, att_fren_vocab , att_inv_fren_vocab, att_op_max_len, sos_index, eos_index, 32)   #NSA HARDCODED VALUE
	return text

@app.route("/")
def hello():
	return render_template("first_page.html")

@app.route("/translate_page", methods = ["POST"])
def translate():

	print("---------encoding for transformer---------------")
	input_text = str(request.form["input_text"])
	print(input_text)
	prepro_text = PreprocessSentence(input_text)
	print(prepro_text)
	text = TokenizeSentence(prepro_text, eng_vocab)
	print(text)
	text = PadSentence(text, maxlen = ip_max_len, padding = "post", value = 0)
	print("text to translate {0}".format(text))
	
	print("---------transformer translating----------------")
	transformer_op = transformer(text)

	print("-------------encoding text for attention model---------------")
	#encode text by att vocab
	text = TokenizeSentence(prepro_text, att_eng_vocab)
	print(text)
	text = PadSentence(text, maxlen = att_ip_max_len, padding = "post", value = 0)
	print("text to translate {0}".format(text))

	print("----------------attention model translating-----------------")
	attention_op = attention(text)

	try:
		textblob_op = TextBlob(input_text).translate(to = "fr")
	except:
		textblob_op = "same as input or couldn't translate"

	return render_template("result.html", transformer_placeholder = transformer_op, attention_placeholder = attention_op, google_placeholder = textblob_op)

@app.route("/load_data")
def LoadData():
	global eng_vocab, fren_vocab, inverse_eng_vocab, inverse_fren_vocab, ip_max_len, op_max_len, eng_vocab_len, fren_vocab_len,\
	d_model, d_inner, heads, dropout, d_key, layers, lr
	global attention_model, transformer_model
	global att_eng_vocab, att_fren_vocab, att_ip_max_len, att_op_max_len, att_params, att_inv_eng_vocab, att_inv_fren_vocab


	print("------------------loading params---------------")

	eng_vocab ,fren_vocab, inverse_eng_vocab, inverse_fren_vocab, ip_max_len, op_max_len , d_model, d_inner, heads, dropout, d_key, layers, \
	lr = pickle.load(open("pickle/transformer_data_v3.pkl", "rb"))
	eng_vocab_len = len(eng_vocab)
	fren_vocab_len = len(fren_vocab)

	
	#load attention params
	att_op_max_len, att_inv_fren_vocab , att_inv_eng_vocab, att_ip_max_len, att_fren_vocab, att_eng_vocab = pickle.load(open("pickle/atten_data.pkl", "rb"))
	

	print("--------------------loading models-----------------")

	m = transfomer(ip_max_len = ip_max_len, op_max_len = op_max_len, 
	d_model = d_model, d_inner = d_inner, n_head = heads, dropout =dropout, d_q = d_key, d_k=d_key, d_v = d_key, n_layers = layers, 
	eng_vocab_len = eng_vocab_len , fren_vocab_size = fren_vocab_len, lr = lr)
	_ , transformer_model = m.create_model()
	transformer_model.load_weights("model/transformer_weights_v2.h5")
	transformer_model._make_predict_function()

	#load attention model
	attention_model = get_att_model_2(32, 32, att_ip_max_len, att_op_max_len, len(att_eng_vocab), len(att_fren_vocab), 40, 0.2, "tanh", "relu" , 0.001)
	attention_model.load_weights("model/attention_weights_v1.h5")
	attention_model._make_predict_function()

	return render_template("index.html")

if __name__ == "__main__":

	app.run(host = '0.0.0.0', port = os.environ.get("PORT"))























