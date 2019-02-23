"""
runs pipeline
"""
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau

from data_loading import *
from preprocessing import *
from utils import *
from modules import *
from transformer import *
from attention_model import *

filename = "fra.txt"
data_dir = "data/"
log_dir = "logs/"
embed_dir = "./"
model_dir = "model/"
train_transformer = False
train_att = True

vocab_words = 8000
filters = ""
lower = True
oov_token = "<unk>"

max_seq_len = None
padding_value = 0
padding_type = "post"

total_size = 10000
train_size = 9000

#transformer model
d_model = 128
d_inner = 128
heads = 5
layers = 3
dropout = 0.3
d_key = 64

#attention model
nfa = 64
nsa = 64
dense1 = 64
activation1 = "tanh"
activation2 = "relu"

batch_size = 2
epochs = 3
lr = 0.001

samples = 1
update_freq = "epoch"

print("------------------preparing data----------------")

data = LoadData(data_dir + filename)
eng, fren = Preprocess(data)
eng_sen, fren_sen_out, fren_sen_in = AddTokens(eng, fren)

eng_sen, fren_sen_in, fren_sen_out , eng_vocab, fren_vocab, inverse_eng_vocab, inverse_fren_vocab  = Tokenize(eng_sen, fren_sen_out, 
	fren_sen_in, vocab_words, filters= filters, lower= lower, oov_token= oov_token)

eng_sen, fren_sen_in, fren_sen_out = PadSequences(eng_sen, fren_sen_in, fren_sen_out, max_seq_len, padding = padding_type, value = padding_value)

eng_sen_train, eng_sen_val, fren_sen_train_in, fren_sen_val_in, fren_sen_train_out, fren_sen_val_out = split(eng_sen, fren_sen_in, 
	fren_sen_out, total_size, train_size)


print("\n------------loading models--------------------\n")

if(train_transformer):
    print("---------loading transformer--------")
    m = transfomer(ip_max_len = eng_sen.shape[1] , op_max_len = fren_sen_out.shape[1], 
        d_model = d_model, d_inner = d_inner, n_head = heads, dropout = dropout, d_q = d_key, d_k=d_key, d_v = d_key, n_layers = layers, 
        eng_vocab_len = len(eng_vocab)+1 , fren_vocab_size = len(fren_vocab)+1, lr = lr)
    transformer_trainer , transformer_inferno = m.create_model()

if(train_att):
    print("--------loading attention model------------")
    att_trainer, att_inferno = get_att_model(nfa = nfa, nsa = nsa, ip_max_len = eng_sen.shape[1], op_max_len = fren_sen_out.shape[1], eng_vocab_size = len(eng_vocab)+1,
     fren_vocab_size = len(fren_vocab) + 1, dense1 = dense1, dropout = dropout, activation1 = activation1, activation2 = activation2 , lr = lr)

tb = TensorBoard(log_dir= log_dir, histogram_freq= 0, batch_size= batch_size, write_graph=True, write_grads=False, write_images= True, 
    embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq= 'epoch')
ea = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='min', baseline=None, restore_best_weights=True)
rp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=0)   

if(train_att):
    s0 = c0 = np.zeros((len(eng_sen_train), nsa))
    print("-----training attention model-------\n")
    att_trainer.fit([eng_sen_train, fren_sen_train_out, s0, c0], None, batch_size = batch_size, epochs = epochs, callbacks = [tb, ea, rp])

if(train_transformer):
    print("----------training transformer-----------------\n")
    transformer_trainer.fit([eng_sen_train , fren_sen_train_in ,fren_sen_train_out] , None , 
              validation_data = ([eng_sen_val , fren_sen_val_in ,fren_sen_val_out] , None) ,
              batch_size = batch_size , epochs = epochs, callbacks = [tb, ea, rp])

print("------------------dumping data-----------------\n")

ip_max_len = eng_sen.shape[1]
op_max_len = fren_sen_out.shape[1]
eng_vocab_len = len(eng_vocab) + 1
fren_vocab_len = len(fren_vocab) + 1

all_params = [eng_vocab, fren_vocab, inverse_eng_vocab, inverse_fren_vocab, ip_max_len, op_max_len, eng_vocab_len, fren_vocab_len, 
d_model, d_inner, heads, dropout, d_key, layers, lr,
nfa, nsa]

dumper(all_params, "params")
if(train_transformer):
    transformer_inferno.save_weights(model_dir + "transformer_v1.h5")
if(train_att):
    att_inferno.save_weights(model_dir + "att_v1.h5")

print("----------some samples------------------")

if samples == 0:
    eng = eng_sen_train
    fren = fren_sen_train_out
elif samples == 1:
    eng = eng_sen_val
    fren = fren_sen_val_out

if(train_att):
    print("--------att samples")
    for i in range(10):
        index = np.random.randint(0, 1000)
        print(invertor(eng[index], inverse_eng_vocab))
        print(invertor(fren[index], inverse_fren_vocab))
        print(decode_sentence_attention(att_inferno, np.expand_dims(eng[index] , axis = 0) ,fren_vocab , inverse_fren_vocab, fren_sen_out.shape[1], 
            fren_vocab.get("<sos>") , fren_vocab.get("<eos>"), nsa))
        print("-----")

if(train_transformer):
    print("--------transformer samples")
    for i in range(10):
        index = np.random.randint(0, 1000)
        print(invertor(eng[index], inverse_eng_vocab))
        print(invertor(fren[index], inverse_fren_vocab))
        print(decode_sentence_transformer(transformer_inferno, np.expand_dims(eng[index] , axis = 0) ,fren_vocab , inverse_fren_vocab, fren_sen_out.shape[1], fren_vocab.get("<sos>") 
        	, fren_vocab.get("<eos>")))
        print("-----")


