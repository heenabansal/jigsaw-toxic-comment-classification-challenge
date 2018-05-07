
# coding: utf-8

# In[1]:


from sklearn.metrics import roc_auc_score
from keras.layers import Input, Embedding, SpatialDropout1D,LSTM, Conv1D, GRU,Dense, Masking, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, RepeatVector, Dropout, merge,TimeDistributed, Flatten, Permute, Lambda
from keras.optimizers import Adam 
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.preprocessing import sequence
from keras.layers import concatenate, dot
#from text import Tokenizer
from keras.preprocessing.text import Tokenizer
from keras.layers import Concatenate
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard,LearningRateScheduler
from keras.utils import plot_model,multi_gpu_model
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# In[2]:


import keras
import keras.backend as K
import numpy as np
np.random.seed(1234)  # for reproducibility
import cPickle
import os.path
import sys
import nltk
import re
import time
from keras.utils import plot_model
import pandas as pd
import os
multi_gpu_use = False
trial = False
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="2"
if(multi_gpu_use):
    os.environ["CUDA_VISIBLE_DEVICES"]="1,3"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))


# In[3]:


if(trial):
    word_embedding_size = 50
    adam_learning_rate = 0.001
    weights_file = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    GLOVE_DIR = '/data1/heena.bansal/deep_learning_projects/glove_pretrained_embeddings/'#'../glove.6B/'
    n_test = 10
    glove_file_name = 'glove.6B.50d.txt'
    n_Epochs = 1
    BatchSize = 128
    lstm_size = 64
    dense1_size = 50
else:
    word_embedding_size = 300
    adam_learning_rate = 0.001
    weights_file = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    GLOVE_DIR = '/data1/heena.bansal/deep_learning_projects/glove_pretrained_embeddings/'
    n_test = 10000
    glove_file_name = 'glove.6B.300d.txt'
    n_Epochs = 20
    BatchSize = 512
    lstm_size = 128
    dense1_size = 128
if(multi_gpu_use):
    BatchSize *= 2


# In[4]:


def fit_sequences(total_text, text_lines, oov_symbol='oov'):
    # create the tokenizer
    t = Tokenizer(oov_token=oov_symbol)
    # fit the tokenizer on the documents
    t.fit_on_texts(total_text)
    # summarize what was learned
    #print(t.word_counts)
    #print(t.document_count)
    #print(t.word_index)
    #print(t.word_docs)
    # integer encode documents
    encoded_text = t.texts_to_sequences(text_lines)
    return t, encoded_text


# In[5]:


def pad_everything(encoded_text):
    maxlen_input = max([len(x) for x in encoded_text])
    padded_text = sequence.pad_sequences(encoded_text, maxlen=maxlen_input)
    return padded_text, maxlen_input


# In[6]:


train = pd.read_csv('train.csv',sep=',',usecols=[1,2,3,4,5,6,7])
test = pd.read_csv('test.csv',sep=',')
#fill nas
train["comment_text"] = train["comment_text"].fillna("unknown")
test["comment_text"] = test["comment_text"].fillna("unknown")


# In[7]:


def calculating_class_weights(labels):
    number_dim = np.shape(labels)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0.,1.], labels[:, i])
    return weights


# In[8]:


text_lines = train.comment_text.values
total_text = train.comment_text.values.tolist() + test.comment_text.values.tolist()
tokenizer1, encoded_text = fit_sequences(total_text, text_lines, oov_symbol='oov')
train['comment_text_encoded'] = encoded_text
train['comment_text_encoded_len'] = train.comment_text_encoded.map(lambda x: len(x))
#drop all the rows from training data where the length of text is too long
#train.comment_text_encoded_len.quantile(0.85)
#train.describe()
train = train.loc[train.comment_text_encoded_len <= 100]
padded_text, maxlen_input = pad_everything(train.comment_text_encoded)

#padded_text is out training data
#we need labels now from train 
labels = np.asarray(train.iloc[:,1:7])
#class_weights = 1 - labels.mean(axis=0)
#class_weights = np.asarray(class_weights.values.tolist())
weights = calculating_class_weights(labels)

labels.shape

Xtrain, Ytrain = padded_text, labels
Ytrain.shape


# In[9]:


import gc
del train
del padded_text
del labels
gc.collect()


# In[10]:


vocabulary = tokenizer1.word_index
dictionary_size = len(vocabulary)
print tokenizer1.word_index['oov']
print maxlen_input


# In[ ]:


embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, glove_file_name))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


# In[ ]:


embedding_matrix = np.zeros((dictionary_size, word_embedding_size))
# Using the Glove embedding:
for word in vocabulary:
    embedding_vector = embeddings_index.get(word)
    index = vocabulary[word]
    if (embedding_vector is not None) and (index < dictionary_size):
        embedding_matrix[index] = embedding_vector

del embeddings_index
gc.collect()


# In[11]:


def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss


# In[12]:


class roc_callback(keras.callbacks.Callback):
    def __init__(self,training_data,validation_data):
        
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        
    
    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):        
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)      
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)      
        print('\n\rroc-auc: %s \rvalidation roc: %s' % (str(round(roc,4)),str(round(roc_val,4)))+'\n')   #,str(round(roc_val,4))
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return   


# In[13]:


def get_model(maxlen_input, lstm_size,embedding_matrix,dense1_size):
    inp = Input(shape=(maxlen_input, ))
    x = Embedding(dictionary_size, embedding_matrix.shape[1])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(lstm_size, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
    x = Conv1D(lstm_size/2, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    x = Dropout(0.1)(x)
    #x = Dense(dense1_size, activation="relu")(x)
    #x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    if(multi_gpu_use):
        model = multi_gpu_model(model, gpus=2)
    model.compile(loss=get_weighted_loss(weights),optimizer='adam',
                  metrics=['accuracy'])#,loss='binary_crossentropy')
    return model


# In[ ]:


def step_decay(epoch):
        initial_lrate = 0.001
        drop_every = 10
        lr = (initial_lrate*np.power(0.5, np.floor((1+drop_every)/drop_every))).astype('float32')
        return lr


# In[ ]:


model = get_model(maxlen_input, lstm_size,embedding_matrix,dense1_size)

X_train, X_test, y_train, y_test = train_test_split(Xtrain, Ytrain, test_size=0.1, random_state=123)
checkpoint_callback = ModelCheckpoint(weights_file, monitor='val_loss', verbose=0, 
                           save_best_only=True, mode='auto', period=5)
tensorboard_callback = TensorBoard(log_dir='./logs', batch_size=BatchSize)
roc_call = roc_callback([X_train,y_train],[X_test,y_test])
early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
lrate = LearningRateScheduler(step_decay)
model.fit(X_train, y_train, batch_size=BatchSize, epochs=n_Epochs, validation_data=(X_test,y_test), 
          callbacks=[checkpoint_callback,tensorboard_callback,early,roc_call,lrate])


# In[14]:


from keras.models import load_model, model_from_json
modelFile = 'weights.05-0.33.hdf5'
model = load_model(modelFile, custom_objects={'weighted_loss': get_weighted_loss(weights)})
test_encoded_text = tokenizer1.texts_to_sequences(test.comment_text.values.tolist())
test_padded_text = sequence.pad_sequences(test_encoded_text, maxlen=maxlen_input)
y_test = model.predict(test_padded_text)

sample_submission = pd.read_csv("sample_submission.csv")
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

sample_submission[list_classes] = y_test
sample_submission.to_csv("gru_cnn_baseline.csv", index=False)

