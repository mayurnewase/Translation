from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
#from utils import *
#from modules import *

def get_att_model(nfa ,nsa , ip_max_len , op_max_len , eng_vocab_size, fren_vocab_size, dense1, dropout, activation1, activation2 , lr):

    first_encoder = Bidirectional(layer = LSTM(nfa , return_sequences = True))
    second_encoder = Bidirectional(layer = LSTM(nfa , return_sequences = True))
    
    first_decoder = LSTM(nsa , return_state = True)
    
    last_densor = Dense(fren_vocab_size + 1)
    activator = Activation(custom_softmax_2)
    
    inp = Input(shape = (ip_max_len,))
    out = Input(shape = (op_max_len,))
    s0 = Input(shape = (nsa,))
    c0 = Input(shape = (nsa,))
    s = s0
    c = c0
    logits = None
    activated_outputs = []

    emb = Embedding(input_dim = eng_vocab_size , output_dim = 64 , input_length = ip_max_len)(inp)
    #emb = Dropout(0.3)(emb)

    encoder_output = first_encoder(emb)
    encoder_output = second_encoder(encoder_output)

    attention = OneStepAttention(ip_max_len, dense1= dense1, dropout= dropout , activation1= activation1, activation2= activation2)
    #encoder_output = third_encoder(encoder_output)
    #encoder_output = fourth_encoder(encoder_output)
    init = 0
    for t in range(op_max_len):    #+1 coz of eos in Y_mapped_output
        context = attention(a = encoder_output , s_prev = s)
        s , _ , c = first_decoder(context , initial_state = [s , c])
        
        output = last_densor(s)
        activate_output = activator(output)

        activated_outputs.append(activate_output)

        #----------------------------------
        if(init == 0):
        	init += 1
        	logits = Lambda(lambda x : tf.expand_dims(x, 1))(output) #;print("output shape is", logits.shape)
        else:
        	output = Lambda(lambda x : tf.expand_dims(x, 1))(output)
        	logits = Lambda(lambda x : tf.concat([x[0], x[1]], axis = 1))([logits, output])  #;print("logits shape stacked", logits.shape)

        #----------------------
    loss_layer = Lambda(get_loss, name = "get_loss")([logits, out])
    accuracy_layer = Lambda(get_accuracy, name = "get_accuracy")([logits, out])

    training_model = Model(inputs = [inp , out , s0 , c0] , outputs = loss_layer)
    
    training_model.add_loss([loss_layer])
    training_model.compile(optimizer = Adam(lr = lr, clipvalue = 0.5))

    training_model.metrics_names.append("accuracy")
    training_model.metrics_tensors.append(accuracy_layer)

    inferno_model = Model(inputs = [inp, s0, c0], outputs = activated_outputs)

    
    return training_model, inferno_model