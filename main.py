"""
runs pipeline
"""
from data_loading import *
from preprocessing import *
from utils import *
from modules import *
from transformer import *
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau

filename = "fra.txt"
data_dir = "data/"
log_dir = "logs/"
embed_dir = "./"
model_dir = "model/"

vocab_words = 8000
filters = ""
lower = True
oov_token = "<unk>"

max_seq_len = None
padding_value = 0
padding_type = "post"

total_size = 10000
train_size = 9000

d_model = 128
d_inner = 128
heads = 5
layers = 3
dropout = 0.3
d_key = 64

batch_size = 64
epochs = 3
lr = 0.001

samples = 1
update_freq = "epoch"

data = LoadData(data_dir + filename)
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

tb = TensorBoard(log_dir= log_dir, histogram_freq= 0, batch_size= batch_size, write_graph=True, write_grads=False, write_images= True, 
	embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq= 'epoch')

ea = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='min', baseline=None, restore_best_weights=True)

rp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=0)	

model.fit([eng_sen_train , fren_sen_train_in ,fren_sen_train_out] , None , 
          validation_data = ([eng_sen_val , fren_sen_val_in ,fren_sen_val_out] , None) ,
          batch_size = batch_size , epochs = epochs, callbacks = [tb, ea, rp])

print("------------------dumping data-----------------\n")
ip_max_len = eng_sen.shape[1]
op_max_len = fren_sen_out.shape[1]
eng_vocab_len = len(eng_vocab) + 1
fren_vocab_len = len(fren_vocab) + 1

all_params = [eng_vocab, fren_vocab, inverse_eng_vocab, inverse_fren_vocab, ip_max_len, op_max_len, eng_vocab_len, fren_vocab_len, 
d_model, d_inner, heads, dropout, d_key, layers, lr]

dumper(all_params, "params")
op_model.save_weights(model_dir + "model_v1.h5")

print("----------some samples------------------")

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


