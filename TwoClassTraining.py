# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:10:56 2024

@author: 赵
"""
import numpy as np
import mne
from mne import io
import h5py
import PyQt5
from mne.datasets import sample
from mne.time_frequency import tfr_morlet
from tensorflow.keras.callbacks import Callback
from mne import Epochs, pick_types, events_from_annotations,concatenate_epochs
from mne.channels import make_standard_montage
import tensorflow as tf
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.compat.v1.keras.backend as K   
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ShuffleSplit, cross_val_score,cross_validate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from keras.layers import Dropout, SpatialDropout2D, LSTM, Reshape
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
# tools for plotting confusion matrices
import kerastuner as kt
from kerastuner import HyperParameters, Hyperband
#from tensorflow.keras.tuner import HyperParameters, Hyperband
from matplotlib import pyplot as plt
from keras.models import load_model
import scipy.io as sio
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import scipy.io
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 选择ID为0的GPU   ######GPU的id怎么查  若只有一个GPU就是0 
le=LabelEncoder()


def normalize_data_per_trial(X):
    """
    将每个试验的每个通道的数据归一化到[-1, 1]之间。
    
    参数：
    - X: 形状为 (n_trials, n_channels, n_times) 的EEG数据
    
    返回：
    - 归一化后的数据
    """
    n_trials, n_channels, n_times = X.shape
    normalized_X = np.zeros_like(X)
    
    for trial in range(n_trials):
        for channel in range(n_channels):
            channel_data = X[trial, channel, :]
            min_val = np.min(channel_data)
            max_val = np.max(channel_data)
            # 使用最小-最大归一化将数据缩放到[-1, 1]之间
            if max_val > min_val:  # 确保分母不为零
                normalized_X[trial, channel, :] = 2 * (channel_data - min_val) / (max_val - min_val) - 1
            else:
                normalized_X[trial, channel, :] = channel_data  # 如果max_val等于min_val，数据保持不变
    return normalized_X


def split_epochs(epochs, train_size=0.60, test_size=0.20):
    """
    对Epochs对象进行随机划分。
    
    参数:
    - epochs: Epochs对象或其他可索引的序列。
    - train_size: 训练集的比例 (0.0到1.0之间)。
    - test_size: 测试集的比例 (0.0到1.0之间)。
    
    返回:
    - epochs_train: 训练集
    - epochs_test: 测试集
    - epochs_valid: 验证集
    """
    # 检查train_size和test_size的有效性
    if not (0 < train_size < 1):
        raise ValueError("train_size必须在0到1之间")
    if not (0 < test_size < 1):
        raise ValueError("test_size必须在0到1之间")
    if train_size + test_size >= 1:
        raise ValueError("train_size和test_size之和必须小于1")
    
    # 计算验证集的比例
    valid_size = 1 - train_size - test_size
    
    # 生成随机索引
    indices = np.arange(len(epochs))
    
    # 划分训练集和剩余集（测试集和验证集）
    train_idx, test_valid_idx = train_test_split(indices, train_size=train_size, random_state=42, stratify=None)
    
    # 划分测试集和验证集
    test_idx, valid_idx = train_test_split(test_valid_idx, train_size=test_size / (1 - train_size), random_state=42, stratify=None)
    
    # 基于索引划分Epochs
    epochs_train = epochs[train_idx]
    epochs_test = epochs[test_idx]
    epochs_valid = epochs[valid_idx]

    return epochs_train, epochs_test, epochs_valid

def shuffle_epochs(epochs):
    
    # 生成随机索引
    indices = np.arange(len(epochs))
    np.random.shuffle(indices)
    
    # 基于随机索引打乱Epochs
    shuffled_epochs = epochs[indices]
    
    return shuffled_epochs




def split_and_concatenate_epochs_2(epochs_s1, epochs_s2):
    """
    分别对每个类的epochs进行随机划分并合并。
    
    参数:
    - epochs_s1: 第一类的Epochs对象。
    - epochs_s2: 第二类的Epochs对象。
    
    返回:
    - epochs_train: 合并后的训练集。
    - epochs_test: 合并后的测试集。
    - epochs_valid: 合并后的验证集。
    """
    # 分别对每个类的epochs进行随机划分
    s1_train, s1_test, s1_valid = split_epochs(epochs_s1)
    s2_train, s2_test, s2_valid = split_epochs(epochs_s2)
    
    # 合并训练集、测试集和验证集
    epochs_train = mne.concatenate_epochs([s1_train, s2_train])
    epochs_test = mne.concatenate_epochs([s1_test, s2_test])
    epochs_valid = mne.concatenate_epochs([s1_valid, s2_valid])
    
    ### 打乱数据集
    epochs_train = shuffle_epochs(epochs_train)
    epochs_test = shuffle_epochs(epochs_test)
    epochs_valid = shuffle_epochs(epochs_valid)

    return epochs_train, epochs_test, epochs_valid


def extract_windows_from_epochs(epochs, window_length, step_size, sfreq, num_windows):
    X, Y = [], []
    num_samples_per_window = int(window_length * sfreq)
    
    for i, epoch_data in enumerate(epochs.get_data()):
        y = epochs.events[i, -1]
        for step in range(num_windows):
            start_sample = int(step * step_size * sfreq)
            end_sample = start_sample + num_samples_per_window
            # 确保即使是最后一个窗口，也不会超出epoch的数据范围
            if end_sample <= epoch_data.shape[1]:
                windowed_data = epoch_data[:, start_sample:end_sample]
                X.append(windowed_data)
                Y.append(y)
            else:
                # 如果超出范围，则跳出循环，不再尝试提取更多窗口
                break
    return np.array(X), np.array(Y)




def EEGNet_rz1(nb_classes=2, Chans = 56, Samples =128, 
             dropoutRate = 0.5, kernLength = 256 , F1 = 8, 
             D = 4, F2 = 32, norm_rate = 1, dropoutType = 'Dropout'):
    
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Chans, Samples, 1))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 256),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, DepthwiseConv2D, SeparableConv2D
from keras.layers import Activation, BatchNormalization, AveragePooling2D, Flatten
from keras.layers import Dropout, SpatialDropout2D, LSTM, Reshape
from keras.constraints import max_norm




tf.keras.backend.clear_session()
#%% load data
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # 仅限于需要时，GPU 内存才增长，对所有 GPU 生效
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    # 内存增长设置必须在程序初始时设置
    print(e)
#%% 导入数据
data_path='E:/张老师实验室/复合肢体运动想象/数据/原始信号/32/lzy/after/'
RLLA_file = 'S1_2_56chs_-epo.fif'
LLRA_file='S1_1_56chs_-epo.fif'

CLASS_1=[LLRA_file]
CLASS_2=[RLLA_file]
CLASSNAME1=CLASS_1[0]
CLASSNAME2=CLASS_2[0]       
s1_epochs=mne.read_epochs(data_path+'/'+CLASSNAME1)  
s2_epochs=mne.read_epochs(data_path+'/'+CLASSNAME2)

tmin_crop = -0.3;
s1_epochs = s1_epochs.crop(tmin=tmin_crop)
s2_epochs = s2_epochs.crop(tmin=tmin_crop)
# 划分数据集
epochs_train, epochs_test, epochs_valid = split_and_concatenate_epochs_2(s1_epochs, s2_epochs)
epochs_train=epochs_train.resample(sfreq=128)  
epochs_valid=epochs_valid.resample(sfreq=128)
epochs_test =epochs_test .resample(sfreq=128)
#删除无用的通道
drop_chs = ['AF3','AF4','F5','F3','F1','Fz','F2','F4','F6','P5','P3','P1','Pz','P2','P4','P6','PO3','POz','PO4']
epochs_train=epochs_train.drop_channels(ch_names=drop_chs)  
epochs_valid=epochs_valid.drop_channels(ch_names=drop_chs)
epochs_test =epochs_test.drop_channels(ch_names=drop_chs)

#在循环外进行测试集等的划分
# 设置窗口大小和步长
window_length = 1.5 # 窗口长度（秒）
step_size = 0.1 # 步长（秒）
sfreq =  128  # 从epochs中获取采样频率

# 计算每个epoch中的窗口数量
num_windows_per_epoch = 4;


# 提取窗口数据
X_train, Y_train = extract_windows_from_epochs(epochs_train, window_length, step_size, sfreq,num_windows_per_epoch)
X_valid, Y_valid = extract_windows_from_epochs(epochs_valid, window_length, step_size, sfreq,num_windows_per_epoch)
X_test, Y_test = extract_windows_from_epochs(epochs_test, window_length, step_size, sfreq,num_windows_per_epoch)

# 数据标准化
X_train = normalize_data_per_trial(X_train)
X_valid = normalize_data_per_trial(X_valid)
X_test = normalize_data_per_trial(X_test)

# 转换标签为分类格式
y_train = np_utils.to_categorical(le.fit_transform(Y_train), 2)
y_valid = np_utils.to_categorical(le.transform(Y_valid), 2)
y_test = np_utils.to_categorical(le.transform(Y_test), 2)

# 适应EEGNet模型
x_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
x_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], X_valid.shape[2], 1)
x_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)


model = EEGNet_rz1(nb_classes=2, Chans = 37, Samples = int(192), 
             dropoutRate = 0.25, kernLength = int(96), F1 =4, 
             D = 4, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout')
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=1, mode='min')


model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=4e-4,decay =3e-6),
              metrics=['accuracy'])
with tf.device('/device:GPU:0'):
    # 训练模型
    history = model.fit(x_train, y_train, epochs=200, batch_size= 64,validation_data=(x_valid, y_valid), callbacks=[early_stopping])
    
    
    probs2       = model.predict(x_test)
    preds2       = probs2.argmax(axis = -1)  
    acc_test  = np.mean(preds2 == y_test.argmax(axis=-1))
    print(acc_test)
    
#print(f'Accuracy: {acc_test*100}')
#directory = 'F://laboratory_zmm//MotorImagery//offlineMI//model//zhouyijun//'
#file_name = 'zyj-2class-512.keras'
#save_path = os.path.join(directory, file_name)
#model.save(save_path)


# Plotting the validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['val_accuracy'])
plt.title('Validation Accuracy Over Epochs')
plt.ylim(0, 1) 
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
# Plotting training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
