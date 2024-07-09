import os
import numpy as np
import mne
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Conv2D, DepthwiseConv2D, SeparableConv2D
from tensorflow.keras.layers import Activation, BatchNormalization, AveragePooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.constraints import max_norm
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import pandas as pd


# GPU设置
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 数据标准化函数
def normalize_data_per_trial(X):
    n_trials, n_channels, n_times = X.shape
    normalized_X = np.zeros_like(X)
    for trial in range(n_trials):
        for channel in range(n_channels):
            channel_data = X[trial, channel, :]
            min_val, max_val = np.min(channel_data), np.max(channel_data)
            if max_val > min_val:
                normalized_X[trial, channel, :] = 2 * (channel_data - min_val) / (max_val - min_val) - 1
            else:
                normalized_X[trial, channel, :] = channel_data
    return normalized_X

# 划分数据集函数
def split_epochs(epochs, train_size=0.60, test_size=0.20):
    indices = np.arange(len(epochs))
    train_idx, test_valid_idx = train_test_split(indices, train_size=train_size, random_state=6)
    test_idx, valid_idx = train_test_split(test_valid_idx, train_size=test_size / (1 - train_size), random_state=42)
    return epochs[train_idx], epochs[test_idx], epochs[valid_idx]

# 随机划分数据的epoch
def shuffle_epochs(epochs):
    indices = np.arange(len(epochs))
    np.random.shuffle(indices)
    return epochs[indices]

# 分割数据集 并实现打乱数据
def split_and_concatenate_epochs_2(epochs_s1, epochs_s2):
    s1_train, s1_test, s1_valid = split_epochs(epochs_s1)
    s2_train, s2_test, s2_valid = split_epochs(epochs_s2)
    epochs_train = mne.concatenate_epochs([s1_train, s2_train])
    epochs_test = mne.concatenate_epochs([s1_test, s2_test])
    epochs_valid = mne.concatenate_epochs([s1_valid, s2_valid])
    return shuffle_epochs(epochs_train), shuffle_epochs(epochs_test), shuffle_epochs(epochs_valid)

def extract_windows_from_epochs(epochs, window_length, step_size, sfreq, num_windows):
    X, Y = [], []
    num_samples_per_window = int(window_length * sfreq)
    for i, epoch_data in enumerate(epochs.get_data()):
        y = epochs.events[i, -1]
        for step in range(num_windows):
            start_sample = int(step * step_size * sfreq)
            end_sample = start_sample + num_samples_per_window
            if end_sample <= epoch_data.shape[1]:
                X.append(epoch_data[:, start_sample:end_sample])
                Y.append(y)
            else:
                break
    return np.array(X), np.array(Y)

# 定义模型
def EEGNet_rznorm(nb_classes=2, Chans=56, Samples=128, dropoutRate=0.5, kernLength=256, F1=8, D=4, F2=32, norm_rate=1, dropoutType='Dropout'):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = tf.keras.layers.SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = tf.keras.layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout.')
    
    input1 = Input(shape=(Chans, Samples, 1))
    
    # 第一层卷积和Batch Normalization
    block1 = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    
    # 深度可分离卷积层
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D, depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)
    
    # 第二层可分离卷积和Batch Normalization
    block2 = SeparableConv2D(F2, (1, 256), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)
    
    # 新增Dropout层
    block2 = dropoutType(dropoutRate)(block2)
        
    flatten = Flatten(name='flatten')(block2)
    
    # 新增Dropout层
    flatten = dropoutType(dropoutRate)(flatten)
    
    dense = Dense(nb_classes, name='dense', kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)

def EEGNet_rz1(nb_classes=2, Chans=56, Samples=128, dropoutRate=0.5, kernLength=256, F1=8, D=4, F2=32, norm_rate=1, dropoutType='Dropout'):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = tf.keras.layers.SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = tf.keras.layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout.')
    
    input1 = Input(shape=(Chans, Samples, 1))
    block1 = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D, depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)
    
    block2 = SeparableConv2D(F2, (1, 256), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)
        
    flatten = Flatten(name='flatten')(block2)
    dense = Dense(nb_classes, name='dense', kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)

################################################################################################################################################
# 加载数据
data_path='F:/张老师实验室/复合肢体运动想象/马芮备份文件/datamr/预处理后数据/MI/S1/'
RLLA_file = 'DH1T2_56chs_-epo.fif'
LLRA_file = 'DH1T1_56chs_-epo.fif'

s1_epochs = mne.read_epochs(os.path.join(data_path, LLRA_file))
s2_epochs = mne.read_epochs(os.path.join(data_path, RLLA_file))

# s1_epochs= s1_epochs.filter(0.1, 16, n_jobs=1, fir_design='firwin', skip_by_annotation='edge')
# s2_epochs= s2_epochs.filter(0.1, 16, n_jobs=1, fir_design='firwin', skip_by_annotation='edge')
tmin_crop = -0.3
s1_epochs.crop(tmin=tmin_crop)
s2_epochs.crop(tmin=tmin_crop)

# 划分数据集
epochs_train, epochs_test, epochs_valid = split_and_concatenate_epochs_2(s1_epochs, s2_epochs)
epochs_train.resample(sfreq=128)
epochs_valid.resample(sfreq=128)
epochs_test.resample(sfreq=128)

# drop_chs = ['AF3', 'AF4', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'PO3', 'POz', 'PO4']
# epochs_train.drop_channels(ch_names=drop_chs)
# epochs_valid.drop_channels(ch_names=drop_chs)
# epochs_test.drop_channels(ch_names=drop_chs)

# 设置窗口大小和步长
window_length = 2
step_size = 0.1
sfreq = 128
num_windows_per_epoch = 5
# 提取窗口数据
X_train, Y_train = extract_windows_from_epochs(epochs_train, window_length, step_size, sfreq, num_windows_per_epoch)
X_valid, Y_valid = extract_windows_from_epochs(epochs_valid, window_length, step_size, sfreq, num_windows_per_epoch)
X_test, Y_test = extract_windows_from_epochs(epochs_test, window_length, step_size, sfreq, num_windows_per_epoch)

# 数据标准化
X_train = normalize_data_per_trial(X_train)
X_valid = normalize_data_per_trial(X_valid)
X_test = normalize_data_per_trial(X_test)

# 转换标签为分类格式
le = LabelEncoder()
y_train = to_categorical(le.fit_transform(Y_train), 2)
y_valid = to_categorical(le.transform(Y_valid), 2)
y_test = to_categorical(le.transform(Y_test), 2)

# 适应EEGNet模型
x_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
x_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], X_valid.shape[2], 1)
x_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# 预定义模型
model = EEGNet_rz1(nb_classes=2, Chans=56, Samples=256, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout')
# 计算模型权重
y_train_int = np.array(Y_train, dtype=int)
# 计算类别权重 防止因为类别不均而导致的失败
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_int), y=y_train_int)
class_weights = dict(enumerate(class_weights))

# 当模型在验证集上的损失不再降低时，触发早停机制。
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, min_lr=8e-5)
# 新增的ModelCheckpoint回调，用于保存验证集上loss最低的5个模型
checkpoint_filepath = 'model_checkpoints/'
os.makedirs(checkpoint_filepath, exist_ok=True)

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=200, batch_size=32, validation_data=(x_valid, y_valid), class_weight=class_weights, callbacks=[early_stopping, reduce_lr])

# 评估模型
probs2 = model.predict(x_test)
preds2 = probs2.argmax(axis=-1)
acc_test = np.mean(preds2 == y_test.argmax(axis=-1))
print(f'Accuracy: {acc_test * 100:.2f}%')

# 绘制训练和验证的准确率和损失
plt.figure(figsize=(10, 5))
plt.plot(history.history['val_accuracy'])
plt.title('Validation Accuracy Over Epochs')
plt.ylim(0, 1)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# print(f'Accuracy: {acc_test*100}')
# directory = 'F://'
# file_name = 'zyj-2class-512.keras'
# save_path = os.path.join(directory, file_name)
# model.save(save_path)
## 保存我当前的history

# 保存history对象
# history_df = pd.DataFrame(history.history)
# history_df.to_csv('training_historyXK1.csv', index=False)

# # 加载history对象
# loaded_history_df = pd.read_csv('training_history.csv')
# loaded_history = loaded_history_df.to_dict(orient='list')
