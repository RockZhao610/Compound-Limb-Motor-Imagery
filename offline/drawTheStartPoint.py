# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:57:49 2024
@author: 赵
我想记录不同滑动窗口的结果。
"""
import os
import numpy as np
import mne
from mne import concatenate_epochs
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Flatten, BatchNormalization, AveragePooling2D, DepthwiseConv2D, SeparableConv2D, Conv2D
from tensorflow.keras.constraints import max_norm
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
# Ensure GPU availability
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Define data augmentation method
def augment_data(X, Y, augment_factor=2):
    augmented_X, augmented_Y = [], []
    for _ in range(augment_factor):
        for x, y in zip(X, Y):
            noise = np.random.normal(0, 0.01, x.shape)
            augmented_X.append(x + noise)
            augmented_Y.append(y)
    return np.array(augmented_X), np.array(augmented_Y)

# Normalize EEG data per trial
def normalize_data_per_trial(X):
    n_trials, n_channels, n_times = X.shape
    normalized_X = np.zeros_like(X)
    
    for trial in range(n_trials):
        for channel in range(n_channels):
            channel_data = X[trial, channel, :]
            min_val = np.min(channel_data)
            max_val = np.max(channel_data)
            if max_val > min_val:
                normalized_X[trial, channel, :] = 2 * (channel_data - min_val) / (max_val - min_val) - 1
            else:
                normalized_X[trial, channel, :] = channel_data
    return normalized_X

# Shuffle epochs
def shuffle_epochs(epochs):
    indices = np.arange(len(epochs))
    np.random.shuffle(indices)
    return epochs[indices]

# Split epochs into training and testing sets
def split_epochs(epochs, train_size=0.85, test_size=0.15):
    indices = np.arange(len(epochs))
    train_idx, test_idx = train_test_split(indices, train_size=train_size, random_state=6)
    return shuffle_epochs(epochs[train_idx]), shuffle_epochs(epochs[test_idx])

def extract_windows_from_epochs(epochs, window_length, step_size, sfreq, num_windows, start_offset=0):
    X, Y = [], []
    num_samples_per_window = int(window_length * sfreq)
    
    for i, epoch_data in enumerate(epochs.get_data()):
        y = epochs.events[i, -1]
        for step in range(num_windows):
            start_sample = int(start_offset * sfreq) + int(step * step_size * sfreq)
            end_sample = start_sample + num_samples_per_window
            if end_sample <= epoch_data.shape[1]:
                windowed_data = epoch_data[:, start_sample:end_sample]
                Y.append(y)
                X.append(windowed_data)
            else:
                break
    return np.array(X), np.array(Y)


# Define EEGNet model
def EEGNet(nb_classes=2, Chans=37, Samples=128, dropoutRate=0.5, kernLength=64, F1=8, D=4, F2=32, norm_rate=0.25, dropoutType='Dropout'):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = tf.keras.layers.SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout, passed as a string.')

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
#%%

# Initialize lists to store metrics
precision_scores = []
recall_scores = []
f1_scores = []
accuracy_scores = []

# Main script
# data_path = 'F:/张老师实验室/复合肢体运动想象/数据/跑全部/'
# SUB_name = ['S1', 'S2', 'S3', 'S4', 'S6', 'S6', 'S3']

SUB_name = ['S1']
data_path='E:/张老师实验室/复合肢体运动想象/数据/原始信号/32/luxh/after/'
# data_path='F:/张老师实验室/复合肢体运动想象/数据/xinshebei/zyh'
RLLA_file = 'S1_2_32chs_-epo.fif'
LLRA_file = 'S1_1_32chs_-epo.fif'


all_subs_score1 = np.zeros(shape=(len(SUB_name), 5))
all_subs_Averagescores = np.zeros(shape=(len(SUB_name), 1))
all_subs_Stdcores = np.zeros(shape=(len(SUB_name), 1))

start_offsets = [0, 0.1, 0.2, 0.3, 0.4,0.5];  # 定义不同的起点偏移量（单位：秒）

results = []

for start_offset in start_offsets:
    print(f"Testing start_offset={start_offset}")
    for j in range(len(SUB_name)):
        name = SUB_name[j]

        # Load data
        s1_epochs = mne.read_epochs(data_path + '/' + LLRA_file)
        s2_epochs = mne.read_epochs(data_path + '/' + RLLA_file)
        
        # Preprocess data
        tmin_crop = -0.5
        s1_epochs = s1_epochs.crop(tmin=tmin_crop)
        s2_epochs = s2_epochs.crop(tmin=tmin_crop)
        epochs_all = concatenate_epochs([s1_epochs, s2_epochs])
        epochs_all = epochs_all.resample(sfreq=128)
        random_state = 42
        
        epochs_train, epochs_test = split_epochs(epochs_all)
        
        # Define StratifiedKFold
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        
        cv_scores = []
        best_accuracy = 0.0  # Initialize best accuracy
        
        with tf.device('/device:GPU:0'):
            labels = epochs_train.events[:, -1]
            le = LabelEncoder()
            labels_encoded = le.fit_transform(labels)
            
            window_length = 2.0
            step_size = 0.10
            sfreq = 128
            num_windows_per_epoch = 1
            
            X_test, Y_test = extract_windows_from_epochs(epochs_test, window_length, step_size, sfreq, num_windows_per_epoch, start_offset=start_offset)
            X_test = normalize_data_per_trial(X_test)
            Y_test = to_categorical(le.transform(Y_test), num_classes=2)
            
            fold_indices = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(epochs_train.get_data(), labels_encoded)):
                # Store the indices in the list
                fold_indices.append({
                    'fold': fold + 1,
                    'train_indices': train_idx.tolist(),
                    'val_indices': val_idx.tolist()
                })
                
                train_data, val_data = epochs_train[train_idx], epochs_train[val_idx]
                X_train, Y_train = extract_windows_from_epochs(train_data, window_length, step_size, sfreq, num_windows_per_epoch, start_offset=start_offset)
                X_val, Y_val = extract_windows_from_epochs(val_data, window_length, step_size, sfreq, num_windows_per_epoch, start_offset=start_offset)
                X_train = normalize_data_per_trial(X_train)
                X_val = normalize_data_per_trial(X_val)
                
                Y_val = to_categorical(le.transform(Y_val), num_classes=2)
                Y_train = to_categorical(le.transform(Y_train), num_classes=2)
                
                model = EEGNet(nb_classes=2, Chans=32, Samples=int(128 * window_length), dropoutRate=0.5, kernLength=int(64*2), F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout')
                model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy'])
                
                history = model.fit(X_train, Y_train, epochs=50, batch_size=64, validation_data=(X_val, Y_val))
                
                # 预测测试集
                Y_pred = model.predict(X_test)
                Y_pred_classes = np.argmax(Y_pred, axis=1)
                Y_true_classes = np.argmax(Y_test, axis=1)
                
                # 计算各项指标
                precision = precision_score(Y_true_classes, Y_pred_classes, average='weighted')
                recall = recall_score(Y_true_classes, Y_pred_classes, average='weighted')
                f1 = f1_score(Y_true_classes, Y_pred_classes, average='weighted')
                accuracy = accuracy_score(Y_true_classes, Y_pred_classes)
                
                # 存储结果
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
                accuracy_scores.append(accuracy)
                
                all_subs_score1[j, fold] = accuracy * 100
                cv_scores.append(accuracy * 100)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    print(f'New best model saved with accuracy: {best_accuracy * 100:.2f}%')
                
                print(f'Validation accuracy for fold {fold + 1}: {accuracy * 100:.2f}%')
            
            # 计算每个受试者的平均值和标准差
            average_accuracy = np.mean(cv_scores)
            std_deviation_accuracy = np.std(cv_scores)
            all_subs_Averagescores[j, 0] = average_accuracy
            all_subs_Stdcores[j, 0] = std_deviation_accuracy
            
            print(f'Average validation accuracy for subject {j+1}: {average_accuracy:.2f}% ± {std_deviation_accuracy:.2f}%')

    # 记录每个起点偏移量的结果
    results.append({
        'start_offset': start_offset,
        'average_precision': np.mean(precision_scores),
        'average_recall': np.mean(recall_scores),
        'average_f1': np.mean(f1_scores),
        'average_accuracy': np.mean(accuracy_scores),
        'std_deviation_accuracy': np.std(accuracy_scores)
    })

# 打印所有起点偏移量的结果
# for result in results:
#     print(f"Start Offset: {result['start_offset']}")
#     print(f"Average Precision: {result['average_precision']:.2f}")
#     print(f"Average Recall: {result['average_recall']:.2f}")
#     print(f"Average F1-Score: {result['average_f1']:.2f}")
#     print(f"Average Accuracy: {result['average_accuracy']*100:.2f}% ± {result['std_deviation_accuracy']*100:.2f}%")
# 绘制起点偏移量的准确率图像
save_dir = 'E:/张老师实验室/复合肢体运动想象/数据/原始信号/32/luxh/after' 
start_offsets = [-0.5+result['start_offset'] for result in results]
accuracies = [result['average_accuracy']*100 for result in results]
std_devs = [result['std_deviation_accuracy']*100 for result in results]

plt.figure(figsize=(10, 6))
plt.errorbar(start_offsets, accuracies, yerr=std_devs, fmt='-o', capsize=5)
plt.ylim(0, 100)
plt.xlabel('Start Offset (s)')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy at Different Start Offsets')
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'SlidingWindows_accuracy.png'))
plt.show()

wb = openpyxl.Workbook()
ws = wb.active

# 写入表头
ws.append(['Accuracy'])

# 写入数据
for value in accuracy_scores:
    ws.append([value])
filename ='accuracyStartpoint.xlsx';
file_path = save_dir + '/' + filename 

# 保存工作簿到文件
wb.save(file_path)