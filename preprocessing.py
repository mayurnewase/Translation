"""
Preprocess data

input -> [lines]
output -> [eng sentences] [french sentences]
"""

import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split

def Preprocess(lines):

	english_sen = []
	french_sen = []

	#re_print = re.compile('[^%s]' % re.escape(string.printable))
	#table = str.maketrans('', '', string.punctuation)
	for index, line in enumerate(lines[:160872]):
	    #line = normalize('NFD', line).encode('ascii', 'ignore')
	    line = line.encode('ascii', 'ignore').decode("utf-8").split("\t")
	    line1 = re.sub(r"[^\w+\s]+\s*\W*" , "" , line[0])
	    line2 = re.sub(r"[^\w+\s]+\s*\W*" , "" , line[1])
	    line1 = line1.lower()
	    line2 = line2.lower()
	    #arr = line.split("\t")
	    english_sen.append(line1)
	    french_sen.append(line2)

	return english_sen, french_sen	

def AddTokens(english_sen, french_sen):
	eng_sen = []
	fren_sen_out = []
	fren_sen_in = []

	for eng,fren in zip(english_sen, french_sen):
	    eng_sen.append(eng)
	    fren_sen_out.append(fren + " <eos>")
	    fren_sen_in.append("<sos> " + fren)

	return eng_sen, fren_sen_out, fren_sen_in

def Tokenize(english_sen, french_sen_in, french_sen_out, vocab_words, filters= "", lower= True, oov_token= "<unk>"):
	engTok = Tokenizer(num_words = vocab_words, filters = filters, lower = lower, oov_token = oov_token)
	freTok = Tokenizer(num_words = vocab_words, filters = filters, lower = lower, oov_token = oov_token)
	engTok.fit_on_texts(list(english_sen))
	freTok.fit_on_texts(list(french_sen_in) + list(french_sen_out))
	english_sen = engTok.texts_to_sequences(english_sen)
	french_sen_in = freTok.texts_to_sequences(french_sen_in)
	french_sen_out = freTok.texts_to_sequences(french_sen_out)

	inverse_eng_vocab = dict([(index , word) for word,index in engTok.word_index.items()])
	inverse_fren_vocab = dict([(index , word) for word,index in freTok.word_index.items()])

	return english_sen, french_sen_in, french_sen_out , engTok.word_index, freTok.word_index, inverse_eng_vocab, inverse_fren_vocab

def PadSequences(english_sen, french_sen_in, french_sen_out, maxlen, padding = "post", value = 0):

	english_sen = pad_sequences(english_sen, maxlen= maxlen, padding= padding, value = value)
	french_sen_in = pad_sequences(french_sen_in, maxlen= maxlen, padding= padding, value = value)
	french_sen_out = pad_sequences(french_sen_out, maxlen= maxlen, padding= padding, value = value)

	return english_sen, french_sen_in, french_sen_out


def split(eng_sen, fren_sen_in, fren_sen_out, total_size, train_size):
	all_index = np.arange(0, total_size)
	train_index, test_index = train_test_split(all_index , train_size = train_size , shuffle = True)

	eng_sen_train = [eng_sen[i] for i in train_index]
	eng_sen_val = [eng_sen[i] for i in test_index]
	fren_sen_train_in = [fren_sen_in[i] for i in train_index]
	fren_sen_val_in = [fren_sen_in[i] for i in test_index]
	fren_sen_train_out = [fren_sen_out[i] for i in train_index]
	fren_sen_val_out = [fren_sen_out[i] for i in test_index]


	return eng_sen_train, eng_sen_val, fren_sen_train_in, fren_sen_val_in, fren_sen_train_out, fren_sen_val_out