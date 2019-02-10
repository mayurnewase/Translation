# --------- 
"""
epochs-60 model-128 inner-64 heads 8 layers 3 dropout 0.2 lr 0.01 max_len=None
"""
#-----------

#-------- COPYING FILE data_loading.py--------------

"""
Load/Download data

input -> filename
output -> [all lines]
"""

import os
import subprocess

def LoadData(filename):
	check = os.path.isfile(filename)
	if(not check):
		process = os.popen("wget http://www.manythings.org/anki/fra-eng.zip").read()
		process = os.popen("unzip fra-eng.zip").read()
		filename = "fra.txt"

	file = open(filename, mode = "rt", encoding = "utf-8")
	data = file.read()
	file.close()

	lines = data.split("\n")
	#data = [s.split("\t") for s in lines]
	return lines
#-------- COPYING FILE preprocessing.py--------------

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
#-------- COPYING FILE utils.py--------------

"""
masks
error
metric
pickle
pos embedding
"""

import pickle
import keras.backend as K
import tensorflow as tf
import numpy as np

def padMask(q,k):
    #input -> src_seq -> [1 , seq_len]
    #op-> 1 for non 0 and 0 for 0 -> [seq_len , seq_len]
    ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2,1])
    return mask

def GetSubMask(s):
    #input -> [1 , seq_len]
    #output -> [seq-len , seq_len] -> matrix with 1 in lower triangle and 0 in upper
    len_s = tf.shape(s)[1]
    bs = tf.shape(s)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask

def get_pos_seq(x):
    #give serially increasing numbers to non-zero element.(0 for 0)
    mask = K.cast(K.not_equal(x, 0), 'int32')
    pos = K.cumsum(K.ones_like(x, 'int32'), 1)
    return pos * mask

def getPosEmbeddingMatrix(max_seq_len , d_embed):

    pos_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / d_embed) for j in range(d_embed)] 
            if pos != 0 else np.zeros(d_embed) 
                for pos in range(max_seq_len)
                ])

    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1    

    return pos_enc


def get_loss(args):
    y_pred, y_true = args
    y_true = tf.cast(y_true, 'int32')
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
    loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
    loss = K.mean(loss)
    return loss
        
def get_accuracy(args):
    y_pred, y_true = args
    mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
    corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
    corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
    return K.mean(corr)

def decode_sentence(model, input_seq ,fren_vocab , inv_fren_vocab, op_max_len, sos_index, eos_index):
    tgt_sen = np.zeros((1 , op_max_len))
    tgt_sen[0,0] = sos_index
    eos_index = fren_vocab.get("<eos>")
    final_sen = ""

    for i in range(op_max_len - 1):
        pred = model.predict_on_batch([input_seq , tgt_sen])
        pred_index = np.argmax(pred[0,i,:])
        tgt_sen[0,i+1] = pred_index
        final_sen = final_sen + inv_fren_vocab.get(pred_index) + " "
        if(pred_index == eos_index):
            break
    return final_sen

def invertor(sen, inv_vocab):
    inv = []
    for i in sen:
        try:
            inv.append(inv_vocab[i])
        except:
            break
    return inv

def dumper(file, name):
	pickle.dump(file, open(name + ".pkl", "wb"))
#-------- COPYING FILE modules.py--------------

"""
Model Components
	Multihead attention
	Dot Product attention
	Layer Norm
	Encoder
	Decoder
"""

from keras.models import *
#from utils import *
from keras.layers import *


class DotProductAttention():
    def __init__(self , d_model, dropout):
        self.scalingFactor = np.sqrt(d_model)
        self.dropout = dropout
        
    def __call__(self , q , k , v , mask , type_layer):
        #find attention -> q * k
         #this should be q * k.T -> but maybe axes takes care of it.
        attention_score = Lambda(lambda x : K.batch_dot(x[0] , x[1] , axes = [2,2]), 
                                 name = "get_attention"+type_layer)([q,k])
        
        if mask is not None:
            #if 0 -> make it small , if 1 -> keep it
            mask_after = Lambda(lambda x : (-1e+10)*(1-x), name = "use_mask"+type_layer)(mask)
            attention_score_after_mask = Add(name="add_attscore_mask"+type_layer)([attention_score , mask_after])

        attention_score_softmax = Activation("softmax")(attention_score_after_mask)
        #print("attention_score ,attention_score_masked , attention_score_softmax" , attention_score.get_shape() ,attention_score_after_mask.get_shape() ,attention_score_softmax.get_shape())
        #print("q , k , v" , q.get_shape() , k.get_shape() , v.get_shape())
        #Find head -> attention * value
        attention_score_softmax = Dropout(self.dropout)(attention_score_softmax)
        head = Lambda(lambda x : K.batch_dot(x[0] , x[1]))([attention_score_softmax , v])
        #print("head" , head.get_shape())
        
        return attention_score , attention_score_after_mask , attention_score_softmax , head

class MultiHeadAttention():
    def __init__(self , n_head , d_model , d_q , d_k , d_v , type_layer, dropout):
        self.n_head = n_head
        self.d_model = d_model
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.type_layer = type_layer
        self.dropout = dropout
        
        self.q_layer = Dense(self.n_head * d_q,name="get_q"+type_layer , use_bias = False) 
        self.k_layer = Dense(self.n_head * d_k,name="get_k"+type_layer , use_bias = False)
        self.v_layer = Dense(self.n_head * d_v,name="get_v"+type_layer , use_bias = False)
        
        self.DpAttention = DotProductAttention(d_model, dropout = self.dropout)   #does (q.k),v
        self.final_attention_layer = Dense(d_model,name="final_att_layer"+type_layer)     #concat all heads and give it to this
        self.layerNormaliztion = LayerNormalization(name = "layer_norm"+self.type_layer)   #special crap than batchnorm
        
    def __call__(self , q , k , v , mask = None):
        #q,k,v -> [bs, max_seq_len , d_embd]
        
        q_mat = self.q_layer(q)         #[bs , seq_len , n_head * d_q]
        k_mat = self.k_layer(k)         #[bs , seq_len , n_head * d_k]
        v_mat = self.v_layer(v)         #[bs , seq_len , n_head * d_v]
        
        #now reshape it to  [n_head * bs , seq_len , d_k] -> stack them to give to dpa
        def reshape1(tensor):
            shape = tf.shape(tensor)
            tensor = tf.reshape(tensor , [shape[0] , shape[1] , self.n_head , self.d_k])
            tensor = tf.transpose(tensor , [2 , 0 , 1 , 3])
            tensor = tf.reshape(tensor , [-1, shape[1] , self.d_k])
            return tensor
        
        q_mat_r1 = Lambda(reshape1, name = "reshape_1_q"+self.type_layer)(q_mat)          #[n_head*bs , seq_len , d_k]
        k_mat_r1 = Lambda(reshape1, name = "reshape_1_k"+self.type_layer)(k_mat)
        v_mat_r1 = Lambda(reshape1, name = "reshape_1_v"+self.type_layer)(v_mat)
        print("q k v" , q_mat_r1.get_shape() , k_mat_r1.get_shape() , v_mat_r1.get_shape())
        
        if mask is not None:                     #stacks the mask for stacked heads -> [n_heads , seq_len , seq_len]
            mask_after = Lambda(lambda x : K.repeat_elements(x , self.n_head , 0), name = "repeat_stack_masks"+self.type_layer)(mask)
        
        attention_score , attention_score_after_mask , attention_score_softmax , head = self.DpAttention(q_mat_r1 , k_mat_r1 , 
                                                                                                         v_mat_r1 , 
                                                                                                         mask = mask_after , type_layer=self.type_layer)        
        #print("attention_score , attention_score_softmax , head" , attention_score.get_shape() , attention_score_softmax.get_shape() , head.get_shape())

        #reshape head to [batch_size, len_v, n_head * d_v] -> dpa gives stacked head ->concat stacked to columns
        def reshape2(tensor):
            shape = tf.shape(tensor)
            tensor = tf.reshape(tensor , [self.n_head , -1 , shape[1] , shape[2]])
            tensor = tf.transpose(tensor , [1,2,0,3])
            tensor = tf.reshape(tensor , [-1 , shape[1] , self.n_head * self.d_v])
            return tensor
        
        head_r2 = Lambda(reshape2, name = "resahpe_2_head" +self.type_layer)(head)
        
        #Final layer
        head_final = self.final_attention_layer(head_r2)     #[batch_size, len_v, d_model]
        head_final = Dropout(self.dropout)(head_final)
        #return head_final , attention_score_softmax
        
        #layer normalization -> STUDY THIS AT LAST
        head_normalized = self.layerNormaliztion(head_final)
        
        #Do below only if layernorm is required
        #Add embd and head_final -> residual connection
        head_plus_query = Add(name = "residual_head_q"+self.type_layer)([head_normalized , q])
        
        #return attention_score, attention_score_after_mask,attention_score_softmax,head,head_r2,head_final,head_plus_query,head_normalized
        return head_plus_query , attention_score , attention_score_after_mask , attention_score_softmax


class Encoder:
    def __init__(self,  d_model , d_inner , n_head , d_q , d_k , d_v , n_layers , word_emb , pos_emb, dropout):
        
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.n_layers = n_layers
        self.word_emb = word_emb
        self.pos_emb = pos_emb
        self.dropout = dropout
        
        self.self_att_layer = MultiHeadAttention(n_head , d_model , d_q , d_k , d_v , type_layer = "_enc_self", dropout = self.dropout)
        self.feedforward = FeedForwardLayer(d_model , d_inner, self.dropout)
        
    def __call__(self , src_seq , src_pos):
        src_embed = self.word_emb(src_seq)
        src_pos = self.pos_emb(src_pos)
        src_final = Add(name = "enc_embd_adder")([src_embed , src_pos])
        
        #get pad mask [seq_len , seq_len]
        mask = Lambda(lambda x : padMask(x,x) , name = "encoder_padmask_src_seq")(src_seq)

        #thing = self.self_att_layer(src_final , src_final , src_final , mask = mask)
        head_normalized , attention_score , attention_score_after_mask , attention_score_softmax = self.self_att_layer(src_final , src_final , src_final , mask = mask)

        #apply feedforward
        feed_forward_op = self.feedforward(head_normalized , "_enc")
        
        return [head_normalized, feed_forward_op , 
                attention_score , attention_score_after_mask , attention_score_softmax ,
                mask]


class Decoder():
    def __init__(self , d_model , d_inner , n_head ,  dq , dk , dv , n_layers , word_emb , pos_emb, dropout):
        
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_q = dq
        self.d_k = dk
        self.d_v = dv
        self.n_layers = n_layers
        self.ip_word_emb = word_emb
        self.ip_pos_emb = pos_emb
        self.dropout = dropout
        
        self.self_att_layer = MultiHeadAttention(self.n_head,self.d_model,self.d_q,self.d_k,self.d_v , "_dec_self", dropout = self.dropout)
        self.enc_dec_att_layer = MultiHeadAttention(self.n_head,self.d_model,self.d_q,self.d_k,self.d_v , "_enc_dec", dropout = self.dropout)
        self.feedForward = FeedForwardLayer(self.d_model , self.d_inner, dropout = self.dropout)
        
    def __call__(self , encoder_output , tgt_seq , tgt_pos , src_seq):
        
        tgt_embed = self.ip_word_emb(tgt_seq)
        tgt_pos = self.ip_pos_emb(tgt_pos)
        
        tgt_final = Add(name = "dec_embd_adder")([tgt_embed , tgt_pos])
        #print("tgt_final" , tgt_final.get_shape())
        
        #Mask Game-----------------------------------
        self_pad_mask = Lambda(lambda x : padMask(x , x),name="padmask_tgt_seq")(tgt_seq) #get 0 for 0 , 1 for other -> [len(x), len(x)]
        self_sub_mask = Lambda(GetSubMask,name="submask_tgt_seq")(tgt_seq) #get lower-one triangle
        self_mask = Lambda(lambda x : K.minimum(x[0] , x[1]),name="self_mask")([self_pad_mask, self_sub_mask]) #->get triangle and pad
        enc_mask = Lambda(lambda x : padMask(x[0] , x[1]),name="enc_mask")([tgt_seq , src_seq]) #n_rows(tgt_seq) mask(src_seq) ->[len(tgt_seq) , len(src_seq)]
        #print("self-pad-mask , self-sub-mask" , self_pad_mask.get_shape() , self_sub_mask.get_shape())
        #print("self_mask , enc_mask" , self_mask.get_shape() , enc_mask.get_shape())
        #--------------------------------------------
        head_normalized_self_dec , attention_score_self_dec , attention_score_after_mask_self_dec , attention_score_softmax_self_dec = self.self_att_layer(
            tgt_final , tgt_final , tgt_final , self_mask)
        #print("head_normalized_self,attention_score",head_normalized_self.get_shape() , attention_score_softmax_self.get_shape())
        
        head_normalized_cross_dec , attention_scor_cross_dec , attention_score_after_mask_cross_dec , attention_score_softmax_cross_dec = self.enc_dec_att_layer(
            head_normalized_self_dec , encoder_output, encoder_output, enc_mask)
        #print("head_normalized_enc_dec,attention_score_enc_dec",head_normalized_enc_dec.get_shape() , attention_score_softmax_enc_dec.get_shape())
        
        feed_forward_op = self.feedForward(head_normalized_cross_dec , "_dec")
        
        return [[head_normalized_self_dec , attention_score_self_dec , \
                        attention_score_after_mask_self_dec , attention_score_softmax_self_dec],
                [head_normalized_cross_dec , attention_scor_cross_dec ,\
                 attention_score_after_mask_cross_dec , attention_score_softmax_cross_dec] ,\
               [self_pad_mask , self_sub_mask , self_mask , enc_mask] , 
               [feed_forward_op]]


class FeedForwardLayer():
    def __init__(self , d_model , d_inner, dropout):
        self.dense1 = Conv1D(d_inner , 1 , activation = "relu")
        self.dense2 = Conv1D(d_model , 1)
        self.layernorm = LayerNormalization()
        self.dropout = dropout
        
    def __call__(self , head , type_layer):
        op1 = self.dense1(head)
        op2 = self.dense2(op1)
        
        encoder_op = Add(name = "residual_ff2_head"+type_layer)([op2 , head])
        encoder_op = Dropout(self.dropout)(encoder_op)
        encoder_op = self.layernorm(encoder_op)
        return encoder_op

class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer="ones", trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:] , initializer="zeros", trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    def compute_output_shape(self, input_shape):
        return input_shape
#-------- COPYING FILE transformer.py--------------

from keras.models import *
from keras.layers import *
from keras.optimizers import Adam

#from utils import *
#from modules import *

class transfomer():
    def __init__(self, ip_max_len, op_max_len, d_model, d_inner, n_head, dropout,
                 d_q, d_k, d_v, n_layers, eng_vocab_len, fren_vocab_size, lr):
        
        self.ip_max_len = ip_max_len
        self.op_max_len = op_max_len
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.dq = d_q
        self.dk = d_k
        self.dv = d_v
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        
        self.ip_word_emb = Embedding(input_dim = eng_vocab_len , output_dim = self.d_model, name="word_embd")
        self.op_word_emb = Embedding(input_dim = fren_vocab_size , output_dim = self.d_model , name="output_word_embd")
        
        self.ip_pos_emb = Embedding(input_dim = eng_vocab_len , output_dim = self.d_model , 
                            weights = [getPosEmbeddingMatrix(eng_vocab_len , self.d_model)] , trainable = False,name="pos_embd_ip")
        self.op_pos_emb = Embedding(input_dim = fren_vocab_size , output_dim = self.d_model , 
                            weights = [getPosEmbeddingMatrix(fren_vocab_size , self.d_model)] , trainable = False,name="pos_embd_op")
        
        self.encoder = Encoder(self.d_model , self.d_inner , self.n_head , self.dq , self.dk , self.dv , self.n_layers , 
                              word_emb = self.ip_word_emb , pos_emb = self.ip_pos_emb, dropout = self.dropout)
        self.decoder = Decoder(self.d_model , d_inner , self.n_head ,  self.dq , self.dk , self.dv , self.n_layers , 
                              word_emb = self.op_word_emb , pos_emb = self.op_pos_emb, dropout = self.dropout)
        
        self.ultimate_layer_1 = TimeDistributed(Dense(fren_vocab_size + 1 , name = "ultimate_layer"))
        
    def create_model(self):
        src_seq_ip = Input(shape=(self.ip_max_len,), dtype='int32' , name = "src_input")
        tgt_seq_ip = Input(shape=(self.op_max_len,), dtype='int32' , name = "target_input")
        tgt_seq_op = Input(shape=(self.op_max_len,), dtype='int32' , name = "target_output")

        #src_seq = src_seq_ip
        
        #tgt_seq  = Lambda(lambda x:x[:,:-1] , name = "modify_target_op")(tgt_seq_ip)      #[1,max_seq-1]
        #tgt_true = Lambda(lambda x:x[:,1:], name = "modify_target_input")(tgt_seq_ip)        #[1,max_seq-1]

        src_pos = Lambda(get_pos_seq, name = "get_src_pos")(src_seq_ip)            #generate seq number except 0.->put 0 for all 0
        tgt_pos = Lambda(get_pos_seq, name = "get_target_pos")(tgt_seq_ip)
        
        #Encoder
        encoder_op = self.encoder(src_seq_ip , src_pos)
        #[head_normalized, feed_forward_op ,attention_score , attention_score_after_mask , attention_score_softmax, mask]
        feed_forward_op = encoder_op[1]
        
        #Decoder
        decoder_op = self.decoder(feed_forward_op , tgt_seq_ip , tgt_pos , src_seq_ip)
        #[[head_normalized_self_dec , attention_score_self_dec, attention_score_after_mask_self_dec , attention_score_softmax_self_dec],
        #[head_normalized_cross_dec , attention_scor_cross_dec ,attention_score_after_mask_cross_dec , attention_score_softmax_cross_dec],
        #[self_pad_mask , self_sub_mask , self_mask , enc_mask] ,[feed_forward_op]]
        feed_forward_op_dec = decoder_op[3][0]
        
        ultimate_output_1 = self.ultimate_layer_1(feed_forward_op_dec)
        
        loss_layer = Lambda(get_loss, name = "get_loss")([ultimate_output_1 , tgt_seq_op])
        accuracy_layer = Lambda(get_accuracy,name = "get_accuracy")([ultimate_output_1, tgt_seq_op])
        
        self.model =  Model([src_seq_ip , tgt_seq_ip, tgt_seq_op] , loss_layer)
        #self.model =  Model([src_seq_ip , tgt_seq_ip] , loss_layer)
        self.model.add_loss([loss_layer])
        self.model.compile(optimizer = Adam(lr = self.lr, clipvalue = 0.5))
        self.model.metrics_names.append("acc")
        self.model.metrics_tensors.append(accuracy_layer)
        
        ultimate_activation = Activation("softmax")(ultimate_output_1)
        self.op_model = Model([src_seq_ip , tgt_seq_ip] , ultimate_activation)
        
        #all_op = encoder_op+decoder_op[0]+decoder_op[1]+decoder_op[2]+decoder_op[3]+[tgt_seq]+[tgt_true]
        #self.all_model = Model([src_seq_ip , tgt_seq_ip] , all_op)
        all_attentions = []
        all_attentions.append(encoder_op[4])#encoder_op[4] + decoder_op[0][3] + deoder_op[1][3]
        all_attentions.append(decoder_op[0][3])
        all_attentions.append(decoder_op[1][3])
        #self.att_model = Model([src_seq_ip , tgt_seq_ip] , all_attentions)
        
        return self.model , self.op_model # , self.att_model



filename = "fra.txt"
embed_dir = "./"
do_preprocess =True
lower =True
vocab_words =6000
oov_token ="<unk>"
max_seq_len =None
padding_value =0
padding_type ="post"
filters =""
total_size =160872
train_size =150000
d_model =128
d_inner =64
d_key =64
heads =8
layers =3
dropout =0.2
epochs =60
batch_size =256
lr =0.01
samples =1
log_dir ="logs/"
update_freq ="epochs"


#-------- COPYING MAIN + --------------

"""
runs pipeline
"""
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau

log_dir = "logs/"
update_freq = "epoch"

data = LoadData(filename)
eng, fren = Preprocess(data)
eng_sen, fren_sen_out, fren_sen_in = AddTokens(eng, fren)

eng_sen, fren_sen_in, fren_sen_out , eng_vocab, fren_vocab, inverse_eng_vocab, inverse_fren_vocab  = Tokenize(eng_sen, fren_sen_out, 
	fren_sen_in, vocab_words, filters= filters, lower= lower, oov_token= oov_token)

eng_sen, fren_sen_in, fren_sen_out = PadSequences(eng_sen, fren_sen_in, fren_sen_out, max_seq_len, padding = padding_type, value = padding_value)

eng_sen_train, eng_sen_val, fren_sen_train_in, fren_sen_val_in, fren_sen_train_out, fren_sen_val_out = split(eng_sen, fren_sen_in, 
	fren_sen_out, total_size, train_size)

#(self, ip_max_len = 44, op_max_len = 51, d_model = 128 , d_inner = 64 , n_head = 4 , dropout = 0.0,
#                 d_q=64, d_k=64 ,d_v=64, n_layers = 4 , eng_vocab_len = 0 , fren_vocab_size = 0)
m = transfomer(ip_max_len = eng_sen.shape[1] , op_max_len = fren_sen_out.shape[1], 
	d_model = d_model, d_inner = d_inner, n_head = heads, dropout = dropout, d_q = d_key, d_k=d_key, d_v = d_key, n_layers = layers, 
	eng_vocab_len = len(eng_vocab)+1 , fren_vocab_size = len(fren_vocab)+1, lr = lr)
model , op_model = m.create_model()

print(len(eng_vocab)) ; print(len(fren_vocab))

tb = TensorBoard(log_dir= log_dir, histogram_freq= 0, batch_size= batch_size, write_graph=True, write_grads=False, write_images= True, 
	embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq= 'epoch')

ea = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='min', baseline=None, restore_best_weights=True)

rp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=0)	

model.fit([eng_sen_train , fren_sen_train_in ,fren_sen_train_out] , None , 
          validation_data = ([eng_sen_val , fren_sen_val_in ,fren_sen_val_out] , None) ,
          batch_size = batch_size , epochs = epochs, callbacks = [tb, ea, rp])

if samples == 0:
    eng = eng_sen_train
    fren = fren_sen_train_out
elif samples == 1:
    eng = eng_sen_val
    fren = fren_sen_val_out

for i in range(10):
    index = np.random.randint(0, 1000)
    print(invertor(eng[index], inverse_eng_vocab))
    print(invertor(fren[index], inverse_fren_vocab))
    print(decode_sentence(op_model, np.expand_dims(eng[index] , axis = 0) ,fren_vocab , inverse_fren_vocab, fren_sen_out.shape[1], fren_vocab.get("<sos>") 
    	, fren_vocab.get("<eos>")))
    print("-----")


