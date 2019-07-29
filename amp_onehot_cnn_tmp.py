import numpy as np
import os
import logging
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, Dropout, Flatten,  Activation, Input, concatenate
import tensorflow as tf
from keras.utils import to_categorical
import random as rn
from sklearn.model_selection import train_test_split
from vis.utils import utils
from keras import activations
from vis.visualization import visualize_saliency, visualize_cam

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
os.environ['PYTHONHASHSEED'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
np.random.seed(123)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# session_conf.gpu_options.per_process_gpu_memory_fraction = 0.4
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

maxlength = 200
input_dim = 20
data_dir = './data/'
# data
all_data = np.load(data_dir + "AMPs_pos_neg_onehot_len200.npz")['data']
all_labels = np.load(data_dir + "AMPs_pos_neg_onehot_len200.npz")['labels']
all_labels = to_categorical(all_labels, num_classes=2)
X_train, X_test_a, Y_train, Y_test_a = train_test_split(all_data, all_labels, test_size=0.4, stratify=all_labels, random_state=43)  # train:test:val=8:1:1, 6:2:2
X_test, X_val, Y_test, Y_val = train_test_split(X_test_a, Y_test_a, test_size=0.5, stratify=Y_test_a, random_state=43)

model = Sequential()
model.add(Conv1D(filters=128, kernel_size=7, input_shape=(maxlength, input_dim,), name='conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=5, name='maxpool2'))
model.add(Conv1D(filters=128, kernel_size=7, name='conv3'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=5, name='maxpool4'))
model.add(Flatten(name='flatten5'))
model.add(Dense(64, name='fl6'))
model.add(Dropout(0.5, name='dropout7'))
model.add(Dense(2, activation='sigmoid', name="prediction"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

train_info_record = './record/'
if not os.path.exists(train_info_record):
    os.mkdir(train_info_record)
checkpoint = ModelCheckpoint(filepath=train_info_record + "save_best_model", monitor='val_acc',
                             save_best_only='True', mode='max', save_weights_only=False)
callback_lists = [checkpoint]
history = model.fit(X_train, np.array(Y_train), validation_data=(X_val, Y_val),
                    epochs=5, batch_size=128, shuffle='True', verbose=1, callbacks=callback_lists)

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
f = open(train_info_record + 'AMP_acc_result.txt', 'a')
epochs = range(1, len(history_dict['acc']) + 1)
for i in range(0, len(history_dict['acc'])):
    f.write('epoch:{}, train_loss:{:.4f}, train_acc:{:.2f}, val_loss:{:.4f}, val_acc:{:.2f}'.
            format(i + 1, loss_values[i], acc_values[i], val_loss_values[i], val_acc_values[i]) + '\n')
_, acc = model.evaluate(X_val, Y_val)
print(acc)

# saliency map
trained_model = load_model(train_info_record + "save_best_model")
layer_idx = utils.find_layer_idx(trained_model, 'prediction')
trained_model.layers[layer_idx].activation = activations.linear
new_model = utils.apply_modifications(trained_model)

indices = np.where(Y_test[:, 0] == 1.)[0]
idx = indices[0]

grads = visualize_saliency(new_model, layer_idx, filter_indices=0, seed_input=X_test[idx],
                           backprop_modifier="guided",)
print("1")
penultimate_layer = utils.find_layer_idx(new_model, 'conv3')
grads_cam = visualize_cam(new_model, layer_idx, filter_indices=0,
                          seed_input=X_test[idx], penultimate_layer_idx=penultimate_layer,
                          backprop_modifier="guided")

print("1")