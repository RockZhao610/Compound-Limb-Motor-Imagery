import statistics
import random 
import sys
import getopt
import numpy as np
import tensorflow
from scipy.io import loadmat
import scipy.io as sio
import os  # handy system and path functions
import datetime
import mne
from mne import Epochs, pick_types, events_from_annotations,concatenate_epochs
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
#import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


# Pylsl发送端
from pylsl import StreamInfo, StreamOutlet,StreamInlet, resolve_stream
from scipy.io import savemat
import serial
import time




def extract_windows_from_data(data, window_length, step_size, sfreq, num_windows):
    X = []
    num_samples_per_window = int(window_length * sfreq)
    num_channels = data.shape[1]

    for step in range(num_windows):
        start_sample = int(step * step_size * sfreq)
        end_sample = start_sample + num_samples_per_window
        if end_sample <= data.shape[2]:
            # 在这里，通过在索引中使用np.newaxis来增加一个新的维度
            # 使得每个窗口的形状为(1, channel, window_size)
            windowed_data = data[:, :, start_sample:end_sample][np.newaxis, :]
            X.append(windowed_data)
        else:
            break
    
   
    return np.concatenate(X, axis=0)

def predict_label(samples, model):
    predicted_labels = []  
    for sample in samples:       
        sample  = np.transpose(sample, (1, 2, 0))
        sample = np.expand_dims(sample, axis=0)
        prediction = model.predict(sample,verbose=0)
        predicted_label = np.argmax(prediction, axis=1)[0]
        predicted_labels.append(predicted_label)
        
    print(predicted_labels)   
    unique, counts = np.unique(predicted_labels, return_counts=True)
    
    label_counts = dict(zip(unique, counts))
    if len(set(counts)) == 1:  
        
        final_label = predicted_labels[-1]  
        
    else:
        
        final_label = max(label_counts, key=label_counts.get)  
        
        
    return final_label

def normalize_data_per_trial(X):
    n_trials, n_channels, _ = X.shape
    # 初始化标准化后的数据数组
    normalized_X = np.zeros_like(X)
    
    for trial in range(n_trials):
        for channel in range(n_channels):
            # 提取当前trial下的当前通道的数据
            channel_data = X[trial, channel, :]
            # 计算均值和标准差
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            # 进行标准化
            normalized_X[trial, channel, :] = (channel_data - mean) / std if std > 0 else channel_data
    
    return normalized_X
# 减少命令行输出
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)  # 忽略特定类型的警告
import logging
logging.basicConfig(level=logging.ERROR)  # 只显示错误信息
def send_marker(data, marker_outlet):
    # 获取当前时间作为时间戳
    marker_timestamp = time.time()
    
    # 打印发送数据的信息
    print("***************************************")
    print('Now sending data: ', data)
    print("***************************************")

    # 将数据和时间戳一起发送
    marker_outlet.push_sample([data], marker_timestamp)


           

markerOutPut_info = StreamInfo('MyOutPutStream', 'Markers', 1, 0, 'string', 'myuidw43537')
marker_outlet = StreamOutlet(markerOutPut_info)
##############################################################################
# 初始化试验计数
trial_count = 0
# 假定有固定数量的试验
total_trials = 60
# 模拟在线数据流文件夹路径
data_folder = "D:/cs/zjquse/EEG data/online data/lixuehan/data/"
marker_folder = "D:/cs/zjquse/EEG data/online data/lixuehan/marker/"
# 加载模型
model = load_model('D:/LIXH-2class-421.keras')
# 获得预测结果
markers=[];
final_labels=[];
##############################################################################
# 使用的初始化文件的地址
data_path = 'D:/cs/zjquse/'
temp_file='CZF_LLRA_56chs_-epo.fif'
# 删去fif文件中原有的数据。
epochs=mne.read_epochs(data_path+temp_file,verbose = 0)
One_epochs = epochs.crop(tmin = -0.5,tmax = 3.0) 
for j in range(59):
    One_epochs = One_epochs.drop(0, reason='USER', verbose=0) 
    #删除59个epoch,因为每次删除epoch的时候，数据会集体向前顶，所以这里设置59轮删除，每次都删除第0个
##############################################################################
#进行假在线
##############################################################################
#获得数据
for trial_count in range(1, total_trials + 1):
    start_time = time.time()  # 获取开始时间
    trial_name = str(trial_count)
    print("**************started*************")
    data_filename = f"EEG_data_Trial_{trial_name}.mat"
    marker_filename = f"Marker_{trial_name}.mat"
    data_path = os.path.join(data_folder, data_filename)
    marker_path = os.path.join(marker_folder, marker_filename)
    # 加载.mat文件
    data = loadmat(data_path)
    EEG_OnlineData=data['EEG_data_after']
##############################################################################
    # 将数据填充在新文件中
    I = []
    for j in range(int(1280)):
        #tem = EEG_data_after[j]
        I.insert(j,EEG_OnlineData[j])    
    temp = np.transpose(I)
    TEMP = np.reshape(temp,(1,56,int(512*2.5)))
    #替换文件中的数据
    One_epochs = mne.EpochsArray(TEMP, One_epochs.info,verbose = 0);
    raw_epoch = One_epochs;
###############################################################################
    # 获取原始 Epoch 的data数据
    Original_data = raw_epoch.get_data()
    n_channels, n_times = Original_data.shape[1], Original_data.shape[2]
    
    # 进行五次镜像延拓
    mirrored_data = Original_data.copy()
    for _ in range(5):
        mirrored_data = np.concatenate([
            mirrored_data[:, :, :-1][..., ::-1],  # 左侧镜像
            mirrored_data,                        # 原始数据
            mirrored_data[:, :, 1:][..., ::-1]    # 右侧镜像
            ], axis=2)
    # 创建一个新的 Epochs 对象
    Extended_epochs = mne.EpochsArray(mirrored_data, raw_epoch.info,verbose = 0)
    # 应用滤波
    filtered_epoch = Extended_epochs.filter(l_freq=1, h_freq=30,verbose=0)
    # 提取滤波后数据的中间部分，大小与原始数据一致
    middle_idx_start = (filtered_epoch.get_data().shape[2] - n_times) // 2
    Filter_data = filtered_epoch.get_data()[:, :, middle_idx_start:middle_idx_start + n_times]
    # 创建一个新的 Epochs 对象，包含处理后的数据
    processed_epoch = mne.EpochsArray(Filter_data, raw_epoch.info,verbose = 0)
    New_epoch =processed_epoch;
    #基线校正
    Epoch_after_baseline = New_epoch.apply_baseline(baseline=(0,0.5),verbose = 0)
    #重参考
    Re_reference_Data = Epoch_after_baseline.copy().set_eeg_reference(ref_channels='average',verbose = 0) 
    #降采样到128Hz
    DownSampling_Data=Re_reference_Data.resample(sfreq=128,verbose = 0);
    #现在删除前0.5s的数据
    Ready_Data =  DownSampling_Data.crop(0.2,2.4)
    # 删去用不上的通道
    drop_chs = ['AF3','AF4','F5','F3','F1','Fz','F2','F4','F6','P5','P3','P1','Pz','P2','P4','P6','PO3','POz','PO4']
    DropReady_Data=Ready_Data.drop_channels(ch_names=drop_chs)  #删除无用的通道
    # 获得最终数据
    Prepro_data =  DropReady_Data.get_data();
    Preproed_data=normalize_data_per_trial(Prepro_data);
    # 预处理结束
################################################################################################################################
    # 获取标签数据
    marker_event = loadmat(marker_path)
    EEG_marker_value =  marker_event['EEG_marker']
    print("makrer:",EEG_marker_value[0])
    markers.append(EEG_marker_value)
################################################################################################################################
    #现在开始模型预测
    print("**************predicting*************")
    # 设置窗口大小和步长
    window_length = 1.0  # 窗口长度（秒）
    step_size = 0.25 # 步长（秒）
    sfreq=128  # 
    # 计算窗口数量
    num_windows_per_trial =5;
    Data_slideWindows=extract_windows_from_data(Preproed_data, window_length, step_size, sfreq,num_windows_per_trial)
    final_label = predict_label (Data_slideWindows, model);
    final_labels.append(final_label);
    # 计算准确率
    # 创建一个映射字典
    markers_mapping = {"Stand": 2, "RLLA": 1, "LLRA": 0}
    # 将 markers 中的字符串转换为对应的数字
    markers_numeric = [markers_mapping[marker[0]] for marker in markers]
    # 创建一个反向映射字典
    # 将 final-labels 中的字符串转换为对应的数字
    reverse_markers_mapping = {value: key for key, value in markers_mapping.items()}
    final_labels_str = [reverse_markers_mapping[final_label]]
    #   计算准确率
    accuracy = sum(final_label == marker for final_label, marker in zip(final_labels, markers_numeric)) / len(markers) * 100
    print("GroudTruth-makrer:",EEG_marker_value[0])
    print("Predicted-makrer:",final_labels_str[0])
    print("准确率:", accuracy)
    end_time = time.time()
    elapsed_time = end_time - start_time  # 计算运行时间
    
    
    
    
    print(f"The code block took {elapsed_time} seconds to execute.")
    send_marker(final_labels_str[0], marker_outlet)
    time.sleep(6)
    print("*************finished*************")
    
   
    
    
