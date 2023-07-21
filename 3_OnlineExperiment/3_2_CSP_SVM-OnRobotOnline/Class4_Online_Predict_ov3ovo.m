% For online prediction of the current MI type.
%% Log
% Updated Time:     2021/11/22
% Author:           Xiangyu Sun(孙翔宇); Chutian Zhang(张楚天); Shixin Ren(任士鑫)
% Version:          V 1.0.0
% File:             Online_Predict.mat
% Describe:         Write during the PhD work at CASIA, Github link: https://github.com/JodyZZZ/MI-BCI
%% Dependent ToolBox：
% 
%% Parameters
% sample_frequency = 256;
% WindowLength = 512;
% 
% channels = [11:14,16:20,22:25];% new cap all channels
% %     Wband = [[1,6];[6,11];[11,16];[16,21];[21,26];[26,31];[31,36]];
% %     Wband = [[10,12];[12,30]];
% %     Wband = [[7,9];[8,10];[9,11];[10,12];[11,13];[12,14];[13,15]];         % alpha / 7
% %     Wband = [[4,8];[8,12];[8,13];[12,16];[13,20];[16,20];[20,24];[24,28];[28,32];[32,36];[36,40]];
% Wband = [[7,9];[8,10];[9,11];[10,12];[11,13];[12,14];[13,15];...
% [4,8];[8,13];[12,16];[13,20];[16,20];[20,24];[24,28];[28,32];[32,36];[36,40]]; % alpha /7
load('InitializationParameters.mat');

%% load the model
disp('	在线模型预测中...')
load('ov3ovo_svm_models.mat');

%% use old model to identify the new observation
% global TrialData;
% rawdata = double(TrialData);
% size_Y = size(rawdata,2);
% rawSlideData_withtag = rawdata(:,(size_Y - WindowLength + 1):size_Y); %取最后2秒数据进行判定

global rawSlideData_withtag;
global PredictType;      

if max(max(rawSlideData_withtag(channels,:)')) > 200
    % in casia lab, without hat attached to amplifier, min is about 114.3415
    PredictType = 0;
else
    [Feas, Y] = Class4_Online_Main_DataPreprocess_ov3ovo(rawSlideData_withtag,sample_frequency,channels,Wband); 
    fprintf('	新样本特征提取... ');
    toc;

    PredictType = svmPredict_4Class_ov3ovo_selectfband(Feas, Y, svmModels);
    fprintf('	新样本类别预测... ');
    toc;
end