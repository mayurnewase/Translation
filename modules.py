"""
Model Components
	Multihead attention
	Dot Product attention
	Layer Norm
	Encoder
	Decoder
"""

from keras.models import *
from keras.layers import *

from utils import *

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


class OneStepAttention():
    def __init__(self, ip_max_len, dense1= 40, dropout= 0.2 , activation1= "tanh", activation2= "relu"):
        self.repeator = RepeatVector(ip_max_len)
        self.concatenator = Concatenate(axis=-1)
        self.densor1 = Dense(dense1, activation = activation1)
        self.dropper = Dropout(dropout)
        self.densor2 = Dense(1, activation = activation2)
        self.activator = Activation(custom_softmax_2 , name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
        self.dotor = Dot(axes = 1)

    def __call__(self, a, s_prev):
        s_prev = self.repeator(inputs = s_prev)
        concat = self.concatenator([a , s_prev])
        e = self.densor1(concat)
        e = self.dropper(e)
        energies = self.densor2(e)
        alphas = self.activator(energies)
        context = self.dotor([alphas , a])
        
        return context


def custom_softmax_2(x, axis=1):
    
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')