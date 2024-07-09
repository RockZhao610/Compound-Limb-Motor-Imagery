%%
clc;
clear all
filename = {'cjq_Session'};
filepath = 'E:\张老师实验室\复合肢体运动想象\数据\原始信号\32\cjq\';
%step1:转换数据
eeglab;
close;
n_sessions=1;
for i=1:n_sessions
    new_filename=char(strcat(filename,num2str(i),'.hdf5'));
    EEG=pop_loadhdf5('filename',new_filename,'filepath',filepath);
    EEG  = pop_creabasiceventlist( EEG , 'AlphanumericCleaning', 'on', 'BoundaryNumeric', { -99 }, 'BoundaryString', { 'boundary' } ); 
    %删除session开始前的数据和结束后的数据
    EEG  = pop_erplabDeleteTimeSegments( EEG , 'displayEEG', 1, 'endEventcodeBufferMS', 2000, 'ignoreUseEventcodes', [1,2,3,4,5,6], 'ignoreUseType','use','startEventcodeBufferMS', 2000, 'timeThresholdMS',  20000);
    save_name=char(strcat(filename,num2str(i),'.set'));
    save_path=char(strcat('E:\张老师实验室\复合肢体运动想象\数据\原始信号\32\cjq\after\',num2str(i)));
    EEG = pop_saveset( EEG, 'filename',save_name,'filepath',save_path);
end
close all;

%% 

%******************************************************************%
%step2:通道定位
clear all;
n_sessions=1;
filename={'cjq_Session'};
filepath='E:\张老师实验室\复合肢体运动想象\数据\原始信号\32\cjq\after\';
 eeglab;
for i=1:n_sessions
    new_filename=char(strcat(filename,num2str(i),'.set'));
    path=char(strcat(filepath,num2str(i)));
    EEG=pop_loadset('filename',new_filename,'filepath',path);
    EEG=pop_chanedit(EEG, 'load',{'E:\张老师实验室\复合肢体运动想象\eeglab\MI_eeglab\gtec_33_channels.loc','filetype','autodetect'});
    EEG = pop_select( EEG, 'nochannel',{'P10'});%删除不用的通道
    %EEG=pop_chanedit(EEG, 'load',{'D:\laboratory_zmm\MotorImagery\offlineMI\MI_eeglab\NEW_68chs(EMG+EOG).ced','filetype','autodetect'}); %67chs: 59EEG(包含2个参考电极）+4EOG+4EMG; 63chs: 57EEG+4EOG; 68chs: 59EEG(包含2个参考电极）+5EOG+4EMG
    %EEG = pop_select( EEG, 'nochannel',{'FP1','REF','AFz','EMG1','EMG2','EMG3','EMG4','LEOG','REOG','HEOG','DEOG','EOG1'});%删除不用的通道
    save_name=char(strcat(filename,num2str(i),'_chanloc','.set'));
    EEG = pop_saveset( EEG, 'filename',save_name,'filepath',path);
end
disp("finish_Step2");
%% 

%************************************批预处理*******************************%
clc;
clear all;
n_sessions=1;
filename={'cjq_Session'};
basepath='E:\张老师实验室\复合肢体运动想象\数据\原始信号\32\cjq\after\';
eeglab;
for i=1:n_sessions
    filepath = fullfile(basepath, num2str(i));
    new_filename=char(strcat(filename,num2str(i),'_chanloc.set'));
    path=char(strcat(filepath,num2str(i)));
    EEG=pop_loadset('filename',new_filename,'filepath',filepath);
    %step3: band-pass filter 1-30Hz
    EEG=pop_eegfiltnew(EEG, 'locutoff',1,'plotfreqz',0);
    EEG=pop_eegfiltnew(EEG, 'hicutoff',30,'plotfreqz',0);
    EEG.setname=char(strcat('cjq_Session',num2str(i),'_bandpass'));

    % %去除线噪声
    % EEG = pop_cleanline(EEG, 'bandwidth',2,'chanlist',[1:56] ,'computepower',1,'linefreqs',60,'newversion',0,'normSpectrum',0,'p',0.01,'pad',2,'plotfigures',0,'scanforlines',0,'sigtype','Channels','taperbandwidth',2,'tau',100,'verb',1,'winsize',4,'winstep',1);
    % new_filename1=char(strcat(filename,num2str(i),'_chanloc_BPfilter.set'));
    % %%插值坏导
    % [clean,~,~,ch_removed] = clean_artifacts(EEG,'Windowcriterion','off');
    % bad=find(ch_removed==1);
    % EEG= pop_interp(EEG, bad, 'spherical');
    % InterpEEG=EEG.data;
    % EEG=pop_loadset('filename',new_filename1,'filepath',filepath);
    % EEG.data(1:56,:)=InterpEEG;%根据实验所需EEG通道重设

    % new_filename2=char(strcat(filename,num2str(i),'_chanloc_BPfilter_ComREF.set'));
    % EEG.setname=char(strcat('S1_Session',num2str(i),'_ASR'));
    % EEG=pop_saveset(EEG,'filename',new_filename2,'filepath',filepath);
    %step 5: common reference
    EEG = pop_reref( EEG, [] );%做共平均参考时一定保证已去除两个双耳参考通道
    new_filename3=char(strcat(filename,num2str(i),'_chanloc_BPfilter_ComREF.set'));
    EEG=pop_saveset(EEG,'filename',new_filename3,'filepath',filepath);
    %STEP 6 DOWN SAMPLE
    %EEG = pop_resample( EEG, 512);%降采样
    new_filename4=char(strcat(filename,num2str(i),'_chanloc_BPfilter_ComREF_DS','.set'));
    EEG = pop_saveset( EEG, 'filename',new_filename4,'filepath',filepath);
%     EEG = pop_runica(EEG, 'icatype', 'jader','chanind',[1:56]);
%     new_filename5=char(strcat(filename,num2str(i),'_chanloc_BPfilter_ASR_ComREF_RunICA','.set'));
%     EEG = pop_saveset( EEG, 'filename',new_filename5,'filepath',path);
end
disp("finish_Step_");
%% 


%************************************提取每个session内的试次*******************************%
clc;
clear all;
n_sessions=1;
filename={'cjq_Session'};
basepath='E:\张老师实验室\复合肢体运动想象\数据\原始信号\32\cjq\after\';
for i=1:n_sessions
    filepath = fullfile(basepath, num2str(i));
    new_filename=char(strcat(filename,num2str(i),'_chanloc_BPfilter_ComREF_DS','.set'));  
    EEG=pop_loadset('filename',new_filename,'filepath',filepath);
    EEG  = pop_creabasiceventlist( EEG , 'AlphanumericCleaning', 'on', 'BoundaryNumeric', { -99 }, 'BoundaryString', { 'boundary' } ); %ERPLAB中的函数

    EEG  = pop_binlister( EEG , 'BDF', 'E:\张老师实验室\复合肢体运动想象\others\trigger\binlister_twoctionsnew.txt', 'IndexEL',  1, 'SendEL2', 'EEG', 'Voutput', 'EEG' ); % GUI: 18-Oct-2021 19:44:02

    EEG = pop_epochbin( EEG , [-1000.0  3000.0],  'none');%marker前时间与marker后时间，稍微设长一点，方便后续用
    new_filename=char(strcat(filename,num2str(i),'_chanloc_BPfilter_ComREF_DS_binlister_extractepochs.set'));
    EEG = pop_saveset( EEG, 'filename',new_filename,'filepath',filepath);
end
disp("finish_Step4");
%% 

% %************************************提取所有session内某一类别的所有试次*******************************%
% %首先合并所有已提取了session内试次的数据集
% clear all;
% eeglab
% filename={'LUXH_session'};
% basepath='E:\张老师实验室\复合肢体运动想象\数据\原始信号\32\luxh\after\';
% n_sessions=1;
% for i=1:n_sessions
%     filepath = fullfile(basepath, num2str(i));
%     new_filename=char(strcat(filename,num2str(i),'_chanloc_BPfilter_ComREF_DS_binlister_extractepochs.set'));
%     EEG=pop_loadset('filename',new_filename,'filepath',filepath);
%     [ALLEEG,EEG,CURRENTSET] = eeg_store(ALLEEG,EEG);
%     disp("finish_"+num2str(i));
% end
% EEG = pop_mergeset( ALLEEG, 1:3, 1);
% EEG.setname='LUXH Merged datasets';
% set_name='LUXH_Merged_Epochs.set';
% EEG = pop_saveset( EEG, 'filename',set_name,'filepath',basepath);
% disp("finish");
%% 
% 打印事件类型
if isfield(EEG, 'event') && ~isempty(EEG.event)
    disp('事件类型列表：');
    % 获取并显示所有事件类型
    eventTypes = {EEG.event.type};
    uniqueEventTypes = unique(eventTypes);
    disp(uniqueEventTypes);
else
    disp('EEG.event 结构体不存在或为空。');
end
%% 

%提取不同类别试次
%clear all;
%eeglab
filename={'cjq_Session'};
filepath='E:\张老师实验室\复合肢体运动想象\数据\原始信号\32\cjq\after\';
Trigger_name={'B1(Trigger1)' ,'B2(Trigger2)' };
Set_name={'LLRA','RLLA'};
Subject_name='cjq';
for i = 1:2
    % 假设这里已经有了合适的EEG变量
    EEG2 = pop_epoch(EEG, Trigger_name(i), [-0.5  3.0], 'newname', char(strcat(Subject_name, Set_name(i))), 'epochinfo', 'yes');
    EEG2 = pop_rmbase(EEG2, [-500 0], []); % 基线校准
    % 注意这里使用 Set_name(i) 来指定每个条件的文件名
    EEG2 = pop_saveset(EEG2, 'filename', [Subject_name '_' char(Set_name(i)) '.set'], 'filepath', filepath);
end
disp("ALLfinish");