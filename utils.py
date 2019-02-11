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
	pickle.dump(file, open("pickle/" + name + ".pkl", "wb"))
