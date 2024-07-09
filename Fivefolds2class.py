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
from tensorflow.keras.callbacks import CSVLogger
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
def split_epochs(epochs, train_size=0.80, test_size=0.20):
    indices = np.arange(len(epochs))
    train_idx, test_idx = train_test_split(indices, train_size=train_size, random_state=6)
    return shuffle_epochs(epochs[train_idx]), shuffle_epochs(epochs[test_idx])

# Extract windows from epochs
def extract_windows_from_epochs(epochs, window_length, step_size, sfreq, num_windows):
    X, Y = [], []
    num_samples_per_window = int(window_length * sfreq)
    
    for i, epoch_data in enumerate(epochs.get_data()):
        y = epochs.events[i, -1]
        for step in range(num_windows):
            start_sample = int(step * step_size * sfreq)
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

SUB_name = ['lzy']
data_path='E:/张老师实验室/复合肢体运动想象/数据/原始信号/32/lzy/after/'
# data_path='F:/张老师实验室/复合肢体运动想象/数据/xinshebei/zyh'
RLLA_file = 'S1_2_32chs_-epo.fif'
LLRA_file = 'S1_1_32chs_-epo.fif'


all_subs_score1 = np.zeros(shape=(len(SUB_name), 5))
all_subs_Averagescores = np.zeros(shape=(len(SUB_name), 1))
all_subs_Stdcores = np.zeros(shape=(len(SUB_name), 1))

for j in range(len(SUB_name)):
    name = SUB_name[j]

    # Load data
    s1_epochs = mne.read_epochs(data_path + '/' + LLRA_file)
    s2_epochs = mne.read_epochs(data_path + '/' + RLLA_file)
    # s3_epochs = mne.read_epochs(data_path2 + '/' + '/' + LLRA_file)
    # s4_epochs = mne.read_epochs(data_path2 + '/' + '/' + RLLA_file)
    
    # Preprocess data
    tmin_crop = -0.5
    s1_epochs = s1_epochs.crop(tmin=tmin_crop)
    s2_epochs = s2_epochs.crop(tmin=tmin_crop)
    # s3_epochs = s3_epochs.crop(tmin=tmin_crop)
    # s4_epochs = s4_epochs.crop(tmin=tmin_crop)
    # epochs_all = concatenate_epochs([s1_epochs, s2_epochs,s3_epochs,s4_epochs])
    epochs_all = concatenate_epochs([s1_epochs, s2_epochs])
    # drop_chs = ['AF3','AF4','F5','F3','F1','Fz','F2','F4','F6','P5','P3','P1','Pz','P2','P4','P6','PO3','POz','PO4']
    # epochs_all = epochs_all.drop_channels(ch_names=drop_chs)
    epochs_all = epochs_all.resample(sfreq=128)
    random_state = 42
    
    epochs_train, epochs_test = split_epochs(epochs_all)
    
    # Define StratifiedKFold
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    #early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.80, patience=3, min_lr=8e-5)
    
    cv_scores = []
    best_accuracy = 0.0  # Initialize best accuracy
    best_model_path = 'best_model.keras'  # Model save path
    save_dir = 'E:/张老师实验室/复合肢体运动想象/数据/原始信号/lzy/after'  # 替换为你的保存路径
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with tf.device('/device:GPU:0'):
        labels = epochs_train.events[:, -1]
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        
        window_length = 2.0
        step_size = 0.10
        sfreq = 128
        num_windows_per_epoch = 4
        counter = 1
        
        X_test, Y_test = extract_windows_from_epochs(epochs_test, window_length, step_size, sfreq, num_windows_per_epoch)
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
            X_train, Y_train = extract_windows_from_epochs(train_data, window_length, step_size, sfreq, num_windows_per_epoch)
            X_val, Y_val = extract_windows_from_epochs(val_data, window_length, step_size, sfreq, num_windows_per_epoch)
            X_train = normalize_data_per_trial(X_train)
            X_val = normalize_data_per_trial(X_val)
            
            Y_val = to_categorical(le.transform(Y_val), num_classes=2)
            Y_train = to_categorical(le.transform(Y_train), num_classes=2)
            
            model = EEGNet(nb_classes=2, Chans=32, Samples=int(128 * window_length), dropoutRate=0.5, kernLength=int(64*2), F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout')
            model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy'])
            
            # history = model.fit(X_train, Y_train, epochs=300, batch_size=32, validation_data=(X_val, Y_val), callbacks=[early_stopping, reduce_lr])
            
            # # 预测测试集
            # Y_pred = model.predict(X_test)
            # Y_pred_classes = np.argmax(Y_pred, axis=1)
            # Y_true_classes = np.argmax(Y_test, axis=1)
            csv_logger = CSVLogger(os.path.join(save_dir, f'fold_{fold + 1}_log.csv'))
            history = model.fit(X_train, Y_train, epochs=36, batch_size=32, validation_data=(X_val, Y_val))#, callbacks=[early_stopping, reduce_lr])
            
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
            
            plt.figure(figsize=(10, 5))
            plt.plot(history.history['val_accuracy'])
            plt.title(f'Validation Accuracy Over Epochs - Fold {fold + 1}')
            plt.ylim(0, 1)
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f'fold_{fold + 1}_val_accuracy.png'))
            plt.close()
        
            plt.figure(figsize=(10, 5))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'Training and Validation Loss Over Epochs - Fold {fold + 1}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f'fold_{fold + 1}_loss.png'))
            plt.close()

            # 检查当前模型是否是最优模型
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                model.save(save_dir+'/'+best_model_path)
                print(f'New best model saved with accuracy: {best_accuracy * 100:.2f}%')
            
            print(f'Validation accuracy for fold {fold + 1}: {accuracy * 100:.2f}%')
        
        # 计算每个受试者的平均值和标准差
        average_accuracy = np.mean(cv_scores)
        std_deviation_accuracy = np.std(cv_scores)
        all_subs_Averagescores[j, 0] = average_accuracy
        all_subs_Stdcores[j, 0] = std_deviation_accuracy
        
        print(f'Average validation accuracy for subject {j+1}: {average_accuracy:.2f}% ± {std_deviation_accuracy:.2f}%')

# 计算总体平均值和标准差
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)
average_f1 = np.mean(f1_scores)
average_accuracy = np.mean(accuracy_scores)
std_deviation_accuracy = np.std(accuracy_scores)

print(f'Average precision: {average_precision:.2f}')
print(f'Average recall: {average_recall:.2f}')
print(f'Average F1-score: {average_f1:.2f}')
print(f'Average validation accuracy: {average_accuracy*100:.2f}% ± {std_deviation_accuracy*100:.2f}%')

average_score = np.mean(all_subs_Averagescores)
average_std = np.mean(all_subs_Stdcores)
print(f'The average score is {average_score:.2f}% ± {average_std:.2f}%')
# 创建一个新的工作簿
wb = openpyxl.Workbook()
ws = wb.active

# 写入表头
ws.append(['Accuracy'])

# 写入数据
for value in accuracy_scores:
    ws.append([value])
filename ='accuracy.xlsx';
file_path = save_dir + '/' + filename 

# 保存工作簿到文件
wb.save(file_path)

