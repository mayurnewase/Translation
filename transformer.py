from keras.models import *
from keras.layers import *
from keras.optimizers import Adam

from utils import *
from modules import *

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
        self.eng_vocab_len = eng_vocab_len
        self.fren_vocab_size = fren_vocab_size
        
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


