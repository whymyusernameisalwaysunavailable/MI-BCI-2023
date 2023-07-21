% This script is parfor CSP online 4-class data pre-prossing, model training and testing.
% CSP parfor data pre-processing; KNN parfor model training and testing
% 
% do 9 cross validation twice if necessary
% discard samples if accuracy doesnot increase

%% Log
% Updated Time:     2021/11/23
% Author:           Chutian Zhang(张楚天); Xiangyu Sun(孙翔宇); Shixin Ren(任士鑫)
% Version:          V 1.0.0
% File:             Online_ModelUpdate_Class2.mat
% MATLAB Version:   R2018b
% Describe:         Write during the PhD work at CASIA, Github link: https://github.com/JodyZZZ/MI-BCI

%% Dependent ToolBox：
% tcp_udp_ip_2.0.6

%% Clear all
addpath(genpath('toolbox'));
clc;clear;close all;delete(gcp('nocreate'));
%parpool(4);

diary(FunctionNowFilename('outputlog_ModelUpdate', '.txt' ));
diary on;

%% Establish MatLab_ModelUpdate to MatLab_Main comm.
global MatlabComm_ModelUpdate
bytesToRead_Main = 1;
MatlabComm_ModelUpdate = tcpip('localhost',9991,'NetworkRole','client');   % set TCP/IP conn. client end, PortNum: 9991
MatlabComm_ModelUpdate.InputBuffersize = 10000;
MatlabComm_ModelUpdate.OutputBuffersize = 10000;
MatlabComm_ModelUpdate.BytesAvailableFcn = {@Online_GetMainFeedback,bytesToRead_Main};
MatlabComm_ModelUpdate.BytesAvailableFcnMode = 'byte';
MatlabComm_ModelUpdate.BytesAvailableFcnCount = bytesToRead_Main;
fopen(MatlabComm_ModelUpdate);                                             % Start the server, return until connection estabilished
% get(MatlabComm_ModelUpdate);
pause(0.1);

% MatLab_ModelUpdate to MatLan_Main Protocol
%   Receive:1-byte message:
%       Byte 1: data file update status
global dataReceive_FromMain;
dataReceive_FromMain = [];
%   Send:1-byte message:
%       Byte 1: model file update status
dataSend_ToMain = uint8(0);
fwrite(MatlabComm_ModelUpdate,dataSend_ToMain);
pause(0.1);
[~,MainOutputSize] = size(dataReceive_FromMain);

%% Data Parameters
% sample_frequency = 256;
% WindowLength = 512;                                                        % Window length
% SlideWindowLength = 256;                                                   % Slide length
% windows_per_trial = 3;
% 
% %% Channel Selection
% %     channels = [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]; 
% %     channels = [22:24];% C3, CZ, C4
% % channels = [16:20];% new cap all channels
% channels = [11:14,16:20,22:25];
% 
% %% Sub Fre Analysis
% %     Wband = [[1,6];[6,11];[11,16];[16,21];[21,26];[26,31];[31,36]];
% %     Wband = [[10,12];[12,30]];
% % Wband = [[7,9];[8,10];[9,11];[10,12];[11,13];[12,14];[13,15]]; % alpha /7
% % Wband = [[4,8];[8,12];[8,13];[12,16];[13,20];[16,20];[20,24];[24,28];[28,32];[32,36];[36,40]];
% Wband = [[7,9];[8,10];[9,11];[10,12];[11,13];[12,14];[13,15];...
%     [4,8];[8,13];[12,16];[13,20];[16,20];[20,24];[24,28];[28,32];[32,36];[36,40]]; % alpha /7
% 
% FilterType = 'bandpass';
% FilterOrder = 4;    
% %% CSP feature
% FilterNum = 4;% 不能超过通道数
load('InitializationParameters.mat');
number_of_channels = length(channels);
number_bandpass_filters = size(Wband,1);

% TrialNum = 48;
load('updateinfo.mat');

%% Update model
update_count = 0;
update_trial = 0;
tmp = -1;
while update_trial <= SceneNum*2
    %% load data (updated data from online data update, in cells of 2 sec EEG data)
    data_tmp = dataReceive_FromMain;
    if data_tmp(end) == update_trial && (tmp == 0 || tmp == -1)
        fprintf('This is the used Dataset, no need to update model.\n\n');
        tmp = 1;
	elseif data_tmp(end) == 0 && tmp == 0
        disp('Dataset updating......');
        disp(' ');
        tmp = 1;
    elseif data_tmp(end) > update_trial
        tic;
        tmp = 0;
        update_trial = data_tmp(end);
        update_count = update_count + 1;
        fprintf('---------------------------------------------------------- Update count: %d.\n',update_count);
        fprintf('                                                           Update trial: %d.\n',update_trial);
        disp('Using updated Dataset.');
        disp('New model training:');
        
         %% Generating Model using updated data
        load ('SlideDataUpdate_Idle.mat');
        load ('SlideDataUpdate_Walk.mat');
        load ('SlideDataUpdate_Ascend.mat');
        load ('SlideDataUpdate_Descend.mat');

        SlideSample_Idle = SlideDataUpdate_Idle;
        SlideSample_Walk = SlideDataUpdate_Walk;
        SlideSample_Ascend = SlideDataUpdate_Ascend;
        SlideSample_Descend = SlideDataUpdate_Descend;

        assert(size(SlideSample_Idle,2) == size(SlideSample_Walk,2) ...
             && size(SlideSample_Walk,2) == size(SlideSample_Ascend,2) ...
             && size(SlideSample_Ascend,2) == size(SlideSample_Descend,2) ,...
            'Sample numbers of different actions should be the same.');

        windows_per_action = size(SlideSample_Idle,2); %window_per_action=trials * windows
        
        fprintf('  样本数据标准化...');
        % normalization
        for i = 1:windows_per_action
            signal_idlesample = SlideSample_Idle{1, i};
            signal_idlesample(channels,:) = signal_idlesample(channels,:) - mean(signal_idlesample(channels,:));
            signal_idlesample(1:end-1,:) = signal_idlesample(1:end-1,:)./var(signal_idlesample(1:end-1,:),0,2);
            SlideSample_Idle{1, i} = signal_idlesample;

            signal_walksample = SlideSample_Walk{1, i};
            signal_walksample(channels,:) = signal_walksample(channels,:) - mean(signal_walksample(channels,:));
            signal_walksample(1:end-1,:) = signal_walksample(1:end-1,:)./var(signal_walksample(1:end-1,:),0,2);
            SlideSample_Walk{1, i} = signal_walksample;

            signal_ascendsample = SlideSample_Ascend{1, i};
            signal_ascendsample(channels,:) = signal_ascendsample(channels,:) - mean(signal_ascendsample(channels,:));
            signal_ascendsample(1:end-1,:) = signal_ascendsample(1:end-1,:)./var(signal_ascendsample(1:end-1,:),0,2);
            SlideSample_Ascend{1, i} = signal_ascendsample;

            signal_descendsample = SlideSample_Descend{1, i};
            signal_descendsample(channels,:) = signal_descendsample(channels,:) - mean(signal_descendsample(channels,:));
            signal_descendsample(1:end-1,:) = signal_descendsample(1:end-1,:)./var(signal_descendsample(1:end-1,:),0,2);
            SlideSample_Descend{1, i} = signal_descendsample;
        end
        
        fprintf('  划分子频带...');
        % shape: (number of bandpass filters, windows per action)
        FilterSample_Idle = cell(number_bandpass_filters, windows_per_action); % 2 Filtered Freq Bands * 105 Windows
        FilterSample_Walk = cell(number_bandpass_filters, windows_per_action);
        FilterSample_Ascend = cell(number_bandpass_filters, windows_per_action);
        FilterSample_Descend = cell(number_bandpass_filters, windows_per_action);

        for i = 1:windows_per_action
            for j = 1:number_bandpass_filters
                FilterSample_Idle{j,i} = Rsx_ButterFilter(FilterOrder,Wband(j,:),sample_frequency,FilterType,SlideSample_Idle{1,i}(channels,:),number_of_channels);
                FilterSample_Walk{j,i} = Rsx_ButterFilter(FilterOrder,Wband(j,:),sample_frequency,FilterType,SlideSample_Walk{1,i}(channels,:),number_of_channels);
                FilterSample_Ascend{j,i} = Rsx_ButterFilter(FilterOrder,Wband(j,:),sample_frequency,FilterType,SlideSample_Ascend{1,i}(channels,:),number_of_channels);
                FilterSample_Descend{j,i} = Rsx_ButterFilter(FilterOrder,Wband(j,:),sample_frequency,FilterType,SlideSample_Descend{1,i}(channels,:),number_of_channels);
            end
        end
        
        samplenum_train = windows_per_action;
        toc;
        
        %% 训练新模型
%         load('ov3ovo_Outcome.mat');
        load('ov3ovo_svm_models.mat');
        load('ov3ovo_CspTranspose_forUse.mat');
        load('updateinfo.mat');

        cmds = cell(5,1);
        cmds{1} = svmModels.cmd_1;
        cmds{2} = svmModels.cmd_2; 
        cmds{3} = svmModels.cmd_3;
        cmds{4} = svmModels.cmd_4;
%         cmds{5} = svmModels.bestg;
%         switch RandomTrial(SceneCount)
%             case 2 % Walking task
%                 excludeType = 4;% not movement type but classifier type
%             case 3 % Ascending task
%                 excludeType = 3;
%             case 4 % Descending task
%                 excludeType = 2;
%         end 

        [CSPs, svmModels] = model4allsample(FilterSample_Idle,FilterSample_Walk,FilterSample_Ascend,FilterSample_Descend,cmds,number_bandpass_filters,FilterNum);
%         [CSPs, svmModels] = model4updatedsample(FilterSample_Idle,FilterSample_Walk,FilterSample_Ascend,FilterSample_Descend,excludeType,cmds,number_bandpass_filters,FilterNum);

        fprintf('    新模型训练完成！\n');

       %% save the model
        disp('Final Decision: ');
        tic;
            
        dataSend_ToMain(1) = hex2dec('01') ;                                   % Notify ModelUpdate dataset is updating
        fwrite(MatlabComm_ModelUpdate,dataSend_ToMain); 

        save('ov3ovo_svm_models.mat', 'svmModels');
        save('ov3ovo_CspTranspose_forUse.mat','CSPs','FilterNum');
        save(FunctionNowFilename('UpdatedModel', '.mat'),...
            'svmModels',...
            'CSPs','FilterNum');

        fprintf('  Trained model Updated.');

        dataSend_ToMain(1) = hex2dec('02') ;                                   % Notify ModelUpdate dataset is updated
        fwrite(MatlabComm_ModelUpdate,dataSend_ToMain);   
       
        toc;
        
        fprintf('Update count: %d.\n',update_count);
        fprintf('Update trial: %d.\n',update_trial);
        fprintf('Trial tag: %d.\n\n',RandomScene(SceneCount));
        
    end

end

% End the session when all trials are completed
disp('This session has ended! Thank you!');
fclose(MatlabComm_ModelUpdate);

diary off;
%delete(gcp('nocreate'));