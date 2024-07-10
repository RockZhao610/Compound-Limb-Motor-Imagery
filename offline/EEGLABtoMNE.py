# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 20:03:21 2022

@author: new
"""
from mne.time_frequency import tfr_morlet


import numpy as np
import mne
import os
import h5py
import scipy.io.matlab as matlab
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from mne.time_frequency import tfr_morlet
import PyQt5
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from mne import Epochs, pick_types, events_from_annotations,concatenate_epochs
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP

#%%
subname=['cjq'] #ME Subjects
#subname=['S1','S2','S3','S4','S5','S6','S7','S8','S9'] #MI Subjects
## 六类任务范式：
##event_name={'LA':1,'RA':2,'LL':3,'RL':4,'LLRA':5,'RLLA':6}
##三类复合肢体任务范式：
event_name={'LLRA':0,'RLLA':1} 
move = ['LLRA','RLLA']
for i in range(1): #受试者人数
    # for ex in range (1):
        for j in range(2): #事件种类
            filename=subname[i]+'_'+move[j]
            # filename=move[j]
            event_number=j+1
            data_path='E:/张老师实验室/复合肢体运动想象/数据/原始信号/32/cjq/after/'
            #data_path='D:/laboratory_zmm/MotorImagery/offlineMI/offlinedata/MR/MI/MI/S1/' 
            data_file=os.path.join(data_path,filename+'.set')
            epochs=mne.read_epochs_eeglab(data_file)
            locs_info_path='E:/张老师实验室/复合肢体运动想象/eeglab/MI_eeglab/gtec_32_channels.loc'
            
            montage=mne.channels.read_custom_montage(locs_info_path)
            #读取正确的导联名称
            new_chan_names=np.loadtxt(locs_info_path,dtype=str,usecols=3)
            #读取旧的导联名称
            old_chan_names=epochs.info["ch_names"]
            #创建字典，匹配新旧导联名称
            chan_names_dict={old_chan_names[i]:new_chan_names[i] for i in range(32)}
            #更新数据中的导联名称
            epochs.rename_channels(chan_names_dict)
            #传入数据的电极位置信息
            epochs.set_montage(montage)
            epochs.events[:,-1]=event_number
            epochs.event_id={key:value for key, value in event_name.items() if value==j+1}
            epochs.save(data_path+'S1'+'_'+str(event_number)+'_32chs_'+'-epo.fif', overwrite=True)
