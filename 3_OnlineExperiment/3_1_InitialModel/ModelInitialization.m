% Model initialization
% generating model based on data collected offline
addpath(genpath('toolbox'));
pnet('closeall');fclose all;clc;clear;close all;
tic;
%% Parameters
% data sample param
sample_frequency = 256;
trials_per_action = 20;                                                    % Trial nummber/action in offline task
seconds_per_TrainTrial = 4;
data_points_per_TrainTrial = sample_frequency * seconds_per_TrainTrial;

WindowLength = 512;                                                        % Window length
SlideWindowLength = 256;                                                   % Slide length
windows_per_trial = (data_points_per_TrainTrial - WindowLength) / SlideWindowLength + 1; % = 3
windows_per_action = windows_per_trial * trials_per_action;                

% channel selection
% new cap 
% Data No.  ... 30   29   28  ...  25     24     23     22   ... 20  19  18  17  16 ...
% Real Pos. ... FC1  FCz  FC2 ... FCC3h  FCC1h  FCC2h  FCC4h ... C3  C1  Cz  C2  C4 ...
% Pos. No.  ... 5    6    7   ...  10     11     12     13   ... 15  16  17  18  19 ...
% Pos. Tag  ... F3   Fz   F4  ... FT11    FC3    FCz    FC4  ... T7  C3  Cz  C4  T8 ... 
%
% Data No.  ...  14     13     12     11   ...  8    7    6  ...
% Real Pos. ... CCP3h  CCP1h  CCP2h  CCP4h ... CP1  CPz  CP2 ...
% Pos. No.  ... 21      22     23     24   ... 27   28   29  ...
% Pos. Tag  ... CPz     CP4    M1     M2   ... Pz   P4   P8  ...
% 
% channels = [16:20]; % 5通道, Cz, C1-C4
% channels = [12:13,17:19,23:24]; % 7通道, Cz, C1, C2, CCP1h, CCP2h, FCC1h, FCC2h
% channels = [11:14,16:20,22:25]; % 13通道, Cz, C1-C4, CCP1h-CCP4h, FCC1h-FCC4h
channels = [6:8,11:14,16:20,22:25,28:30]; % 19通道, Cz, C1-C4, CCP1h-CCP4h, FCC1h-FCC4h, FCz, FC1-FC2, CPz, CP1-CP2

number_of_channels = length(channels);

% sub fre band selection
Wband = [[4,8];[8,13];[13,20];[13,30]];

number_bandpass_filters = size(Wband,1); 
FilterType = 'bandpass';
FilterOrder = 4;   

save('InitializationParameters','sample_frequency',...
                  'trials_per_action','seconds_per_TrainTrial',...
                  'WindowLength','SlideWindowLength',...
                  'channels','Wband',...
                  'FilterType','FilterOrder');

% choose CSP param
FilterNum = 4;% 不能超过通道数

%% Load & save baseline/training data，preprocess w/ slide window，seperate into 4 action types
global TrialData;
load('Offline_EEGdata_20221011_172917981.mat');
TrialDataBase = double(TrialData);

Tag = TrialDataBase(end,:);
% Idle
[~,Index_Idle] = find(Tag==1);
TrialDataUpdate_Idle = double(TrialDataBase(:,Index_Idle));
% Walk
[~,Index_Walk] = find(Tag==2);
TrialDataUpdate_Walk = double(TrialDataBase(:,Index_Walk));
% Ascend
[~,Index_Ascend] = find(Tag==3);
TrialDataUpdate_Ascend = double(TrialDataBase(:,Index_Ascend));
% Descend
[~,Index_Descend] = find(Tag==4);
TrialDataUpdate_Descend = double(TrialDataBase(:,Index_Descend));

% put all 2-sec windows of each action together for further update
SlideDataUpdate_Idle = cell(1, windows_per_action);
SlideDataUpdate_Walk = cell(1, windows_per_action);
SlideDataUpdate_Ascend = cell(1, windows_per_action);
SlideDataUpdate_Descend = cell(1, windows_per_action);

for i = 1:trials_per_action
    for j = 1:windows_per_trial
        PointStart = (i-1)*data_points_per_TrainTrial + (j-1)*SlideWindowLength;
        SlideDataUpdate_Idle{1, (i-1)*windows_per_trial+j} = TrialDataUpdate_Idle(:,PointStart + 1:PointStart + WindowLength );
        SlideDataUpdate_Walk{1, (i-1)*windows_per_trial+j} = TrialDataUpdate_Walk(:,PointStart + 1:PointStart + WindowLength );
        SlideDataUpdate_Ascend{1, (i-1)*windows_per_trial+j} = TrialDataUpdate_Ascend(:,PointStart + 1:PointStart + WindowLength );
        SlideDataUpdate_Descend{1, (i-1)*windows_per_trial+j} = TrialDataUpdate_Descend(:,PointStart + 1:PointStart + WindowLength );
    end
end

SlideDataUpdate_Idle_Base = SlideDataUpdate_Idle;
SlideDataUpdate_Walk_Base = SlideDataUpdate_Walk;
SlideDataUpdate_Ascend_Base = SlideDataUpdate_Ascend;
SlideDataUpdate_Descend_Base = SlideDataUpdate_Descend;

save('SlideDataUpdate_Base.mat',...
    'SlideDataUpdate_Idle_Base','SlideDataUpdate_Walk_Base',...
    'SlideDataUpdate_Ascend_Base','SlideDataUpdate_Descend_Base');
% 保存初始数据，用于观察样本集更新情况
% 每次运行都基于离线数据重新开始更新
 
%% Load Data
SlideSample_Idle = SlideDataUpdate_Idle_Base;
SlideSample_Walk = SlideDataUpdate_Walk_Base;
SlideSample_Ascend = SlideDataUpdate_Ascend_Base;
SlideSample_Descend = SlideDataUpdate_Descend_Base;

assert(size(SlideSample_Idle,2) == size(SlideSample_Walk,2) ...
     && size(SlideSample_Walk,2) == size(SlideSample_Ascend,2) ...
     && size(SlideSample_Ascend,2) == size(SlideSample_Descend,2) ,...
    'Sample numbers of different actions should be the same.');

windows_per_action = size(SlideSample_Idle,2); %window_per_action=trials * windows

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
fprintf('  划分子频带...');
samplenum = windows_per_action;


%% make 3 training-testing pairs of sample sets, 7/10 training, 1/10 testing, according to time
foldnum = 20;
windows_per_fold = windows_per_action/foldnum; % 20 * 3 / 10
% to avoid repetitive feature extraction step 
% which costs most computing time
% while searching optimized [c,g].

for valcount = 1:3
%%% valcount = 1: NO.1-7 training, NO.8 testing
%%% valcount = 2: NO.2-8 training, NO.9 testing
%%% valcount = 3: NO.3-9 training, NO.10 testing
    valindex = foldnum - 3 + valcount;
    FilterSample_Idle_train = FilterSample_Idle(:,(valcount-1)* windows_per_fold + 1:(valindex - 1) * windows_per_fold);
    FilterSample_Walk_train = FilterSample_Walk(:,(valcount-1)* windows_per_fold + 1:(valindex - 1) * windows_per_fold);
    FilterSample_Ascend_train = FilterSample_Ascend(:,(valcount-1)* windows_per_fold + 1:(valindex - 1) * windows_per_fold);
    FilterSample_Descend_train = FilterSample_Descend(:,(valcount-1)* windows_per_fold + 1:(valindex - 1) * windows_per_fold);

    samplenum_train = (foldnum - 3) * windows_per_fold;

    FilterSample_train_Idle = repmat(FilterSample_Idle_train,1,3);
    FilterSample_train_notIdle = [FilterSample_Walk_train,FilterSample_Ascend_train,FilterSample_Descend_train];

    FilterSample_test = [FilterSample_Idle(:,((valindex - 1) * windows_per_fold + 1):valindex * windows_per_fold), ...
                        FilterSample_Walk(:,((valindex - 1) * windows_per_fold + 1):valindex * windows_per_fold), ...
                        FilterSample_Ascend(:,((valindex - 1) * windows_per_fold + 1):valindex * windows_per_fold), ...
                        FilterSample_Descend(:,((valindex - 1) * windows_per_fold + 1):valindex * windows_per_fold)];

    FilterSample_Y_test = [ones(windows_per_fold,1);ones(windows_per_fold,1)*2;...
                        ones(windows_per_fold,1)*3;ones(windows_per_fold,1)*4];


    %% geting CSP
    fprintf('  生成csp矩阵...');
    CspTranspose_1 = cell(1,number_bandpass_filters);
    CspTranspose_23 = cell(1,number_bandpass_filters);
    CspTranspose_24 = cell(1,number_bandpass_filters);
    CspTranspose_34 = cell(1,number_bandpass_filters);
    for i = 1:number_bandpass_filters
        CspTranspose_1{i} = Rsx_CSP_R3(FilterSample_train_Idle(i,:),FilterSample_train_notIdle(i,:)); 
        CspTranspose_23{i} = Rsx_CSP_R3(FilterSample_Walk_train(i,:),FilterSample_Ascend_train(i,:)); 
        CspTranspose_24{i} = Rsx_CSP_R3(FilterSample_Walk_train(i,:),FilterSample_Descend_train(i,:)); 
        CspTranspose_34{i} = Rsx_CSP_R3(FilterSample_Ascend_train(i,:),FilterSample_Descend_train(i,:));  
    end
    CSPs = cell(1,4);
    CSPs{1} = CspTranspose_1;
    CSPs{2} = CspTranspose_23;
    CSPs{3} = CspTranspose_24;
    CSPs{4} = CspTranspose_34;


    %% CSP Feature extraction for generating classifier
    % （只对训练样本进行提取）
    fprintf('  生成训练集特征及标签...');
    % Idle vs. notIdle
    % shape: (samplenum, number of bandpass filters * FilterNum)
    FeaSample_Idle_train_1 = [];
    FeaSample_Walk_train_1 = [];
    FeaSample_Ascend_train_1 = [];
    FeaSample_Descend_train_1 = [];
    for i = 1:samplenum_train %每个window的每个freq band进行特征提取
        FeaTemp_Idle = [];
        FeaTemp_Walk = [];
        FeaTemp_Ascend = [];
        FeaTemp_Descend = [];
        for j =1:number_bandpass_filters
            FeaTemp_Idle = [FeaTemp_Idle,Rsx_singlewindow_cspfeature(FilterSample_Idle_train{j,i},CspTranspose_1{j},FilterNum)];
            FeaTemp_Walk = [FeaTemp_Walk,Rsx_singlewindow_cspfeature(FilterSample_Walk_train{j,i},CspTranspose_1{j},FilterNum)];
            FeaTemp_Ascend = [FeaTemp_Ascend,Rsx_singlewindow_cspfeature(FilterSample_Ascend_train{j,i},CspTranspose_1{j},FilterNum)];
            FeaTemp_Descend = [FeaTemp_Descend,Rsx_singlewindow_cspfeature(FilterSample_Descend_train{j,i},CspTranspose_1{j},FilterNum)];
        end
        FeaSample_Idle_train_1 = [FeaSample_Idle_train_1;FeaTemp_Idle];
        FeaSample_Walk_train_1 = [FeaSample_Walk_train_1;FeaTemp_Walk];
        FeaSample_Ascend_train_1 = [FeaSample_Ascend_train_1;FeaTemp_Ascend];
        FeaSample_Descend_train_1 = [FeaSample_Descend_train_1;FeaTemp_Descend];
    end
    FeaSample_tr_Idle = repmat(FeaSample_Idle_train_1,3,1);
    FeaSample_tr_notIdle = [FeaSample_Walk_train_1; FeaSample_Ascend_train_1; FeaSample_Descend_train_1];

    % Walk vs. Ascend
    % shape: (samplenum, number of bandpass filters * FilterNum)
    FeaSample_Walk_train_23 = [];
    FeaSample_Ascend_train_23 = [];
    for i = 1:samplenum_train %每个window的每个freq band进行特征提取
        FeaTemp_Walk = [];
        FeaTemp_Ascend = [];
        for j =1:number_bandpass_filters
            FeaTemp_Walk = [FeaTemp_Walk,Rsx_singlewindow_cspfeature(FilterSample_Walk_train{j,i},CspTranspose_23{j},FilterNum)];
            FeaTemp_Ascend = [FeaTemp_Ascend,Rsx_singlewindow_cspfeature(FilterSample_Ascend_train{j,i},CspTranspose_23{j},FilterNum)];
        end
        FeaSample_Walk_train_23 = [FeaSample_Walk_train_23;FeaTemp_Walk];
        FeaSample_Ascend_train_23 = [FeaSample_Ascend_train_23;FeaTemp_Ascend];
    end

    % Walk vs. Descend
    % shape: (samplenum, number of bandpass filters * FilterNum)
    FeaSample_Walk_train_24 = [];
    FeaSample_Descend_train_24 = [];
    for i = 1:samplenum_train %每个window的每个freq band进行特征提取
        FeaTemp_Walk = [];
        FeaTemp_Descend = [];
        for j =1:number_bandpass_filters
            FeaTemp_Walk = [FeaTemp_Walk,Rsx_singlewindow_cspfeature(FilterSample_Walk_train{j,i},CspTranspose_24{j},FilterNum)];
            FeaTemp_Descend = [FeaTemp_Descend,Rsx_singlewindow_cspfeature(FilterSample_Descend_train{j,i},CspTranspose_24{j},FilterNum)];
        end
        FeaSample_Walk_train_24 = [FeaSample_Walk_train_24;FeaTemp_Walk];
        FeaSample_Descend_train_24 = [FeaSample_Descend_train_24;FeaTemp_Descend];
    end

    % Ascend vs. Descend
    % shape: (samplenum, number of bandpass filters * FilterNum)
    FeaSample_Ascend_train_34 = [];
    FeaSample_Descend_train_34 = [];
    for i = 1:samplenum_train %每个window的每个freq band进行特征提取
        FeaTemp_Ascend = [];
        FeaTemp_Descend = [];
        for j =1:number_bandpass_filters
            FeaTemp_Ascend = [FeaTemp_Ascend,Rsx_singlewindow_cspfeature(FilterSample_Ascend_train{j,i},CspTranspose_34{j},FilterNum)];
            FeaTemp_Descend = [FeaTemp_Descend,Rsx_singlewindow_cspfeature(FilterSample_Descend_train{j,i},CspTranspose_34{j},FilterNum)];
        end
        FeaSample_Ascend_train_34 = [FeaSample_Ascend_train_34;FeaTemp_Ascend];
        FeaSample_Descend_train_34 = [FeaSample_Descend_train_34;FeaTemp_Descend];
    end

    % Idle vs. notIdle
    train_Fea_1_init = [FeaSample_tr_Idle;FeaSample_tr_notIdle]; 
    train_Fea_Y_1 = [ones(samplenum_train*3,1);ones(samplenum_train*3,1)*2];
    FeaMin_1 = min(train_Fea_1_init);
    FeaMax_1 = max(train_Fea_1_init);
    train_Fea_1 = (train_Fea_1_init - FeaMin_1)./(FeaMax_1-FeaMin_1);
    
    % Walk vs. Ascend
    train_Fea_23_init = [FeaSample_Walk_train_23;FeaSample_Ascend_train_23];
    train_Fea_Y_23 = [ones(samplenum_train,1);ones(samplenum_train,1)*2];
    FeaMin_23 = min(train_Fea_23_init);
    FeaMax_23 = max(train_Fea_23_init);
    train_Fea_23 = (train_Fea_23_init - FeaMin_23)./(FeaMax_23-FeaMin_23);
        
    % Walk vs. Descend
    train_Fea_24_init = [FeaSample_Walk_train_24;FeaSample_Descend_train_24];
    train_Fea_Y_24 = [ones(samplenum_train,1);ones(samplenum_train,1)*2];
    FeaMin_24 = min(train_Fea_24_init);
    FeaMax_24 = max(train_Fea_24_init);
    train_Fea_24 = (train_Fea_24_init - FeaMin_24)./(FeaMax_24-FeaMin_24);

    % Ascend vs. Descend
    train_Fea_34_init = [FeaSample_Ascend_train_34;FeaSample_Descend_train_34];
    train_Fea_Y_34 = [ones(samplenum_train,1);ones(samplenum_train,1)*2];
    FeaMin_34 = min(train_Fea_34_init);
    FeaMax_34 = max(train_Fea_34_init);
    train_Fea_34 = (train_Fea_34_init - FeaMin_34)./(FeaMax_34-FeaMin_34);

    
    %% 生成测试集特征样本及标签
    % CSP Feature extraction (of testing samples)
    fprintf('  生成测试集特征及标签...\n');
    test_Fea_1 = [];
    test_Fea_23 = [];
    test_Fea_24 = [];
    test_Fea_34 = [];
    for i = 1:size(FilterSample_test,2) %每个window的每个freq band进行特征提取
        FeaTemp_1 = [];
        FeaTemp_23 = [];
        FeaTemp_24 = [];
        FeaTemp_34 = [];
        for j =1:number_bandpass_filters
            FeaTemp_1 = [FeaTemp_1,Rsx_singlewindow_cspfeature(FilterSample_test{j,i},CspTranspose_1{j},FilterNum)];
            FeaTemp_23 = [FeaTemp_23,Rsx_singlewindow_cspfeature(FilterSample_test{j,i},CspTranspose_23{j},FilterNum)];
            FeaTemp_24 = [FeaTemp_24,Rsx_singlewindow_cspfeature(FilterSample_test{j,i},CspTranspose_24{j},FilterNum)];
            FeaTemp_34 = [FeaTemp_34,Rsx_singlewindow_cspfeature(FilterSample_test{j,i},CspTranspose_34{j},FilterNum)];
        end
        test_Fea_1 = [test_Fea_1; (FeaTemp_1 - FeaMin_1)./(FeaMax_1 - FeaMin_1)];
        test_Fea_23 = [test_Fea_23; (FeaTemp_23 - FeaMin_23)./(FeaMax_23 - FeaMin_23)];
        test_Fea_24 = [test_Fea_24; (FeaTemp_24 - FeaMin_24)./(FeaMax_24-FeaMin_24)];
        test_Fea_34 = [test_Fea_34; (FeaTemp_34 - FeaMin_34)./(FeaMax_34-FeaMin_34)];
    end
    test_Fea_all = cell(1,4);
    test_Fea_all{1} = test_Fea_1;
    test_Fea_all{2} = test_Fea_23;
    test_Fea_all{3} = test_Fea_24;
    test_Fea_all{4} = test_Fea_34;
    test_Fea_Y = FilterSample_Y_test;

    save(['FeaSets_',num2str(valcount),'_for_classifiers_trte.mat'],...
            'train_Fea_1','train_Fea_Y_1',...
            'train_Fea_23','train_Fea_Y_23',...
            'train_Fea_24','train_Fea_Y_24',...
            'train_Fea_34','train_Fea_Y_34',...
            'test_Fea_all','test_Fea_Y');

end
                       
% toc;


fea_num = number_bandpass_filters * FilterNum;
%% Classifier
fprintf('  pso搜索最佳模型训练参数...');
num_stop_1=0;
num_stop_2=0;
num_stop_3=0;
num_stop_4=0;
num_particle_1=100;
num_particle_2=50;
num_stop = 5000;
% tic;
% CGScope=[-12 12;-12 12];
CGKFbandScope=[1 3;-12 3;1 2^fea_num-0.01;0.3 0.3];
ParticleSize = size(CGKFbandScope,1);
[info_bestclassifier_1,OptSwarm_1,minmax_1,num_stop_1,k_1]=PsoProcessforCGKFband(num_particle_1,ParticleSize,CGKFbandScope,...
                    1,@InitCGSwarm,@BaseStepPso,@CGKFbandperformanceGray,0,0,num_stop,0,num_stop_1,maxOptLowerBound_1,num_stop_UpperBound);

[info_bestclassifier_2,OptSwarm_2,minmax_2,num_stop_2,k_2]=PsoProcessforCGKFband(num_particle_2,ParticleSize,CGKFbandScope,...
                    2,@InitCGSwarm,@BaseStepPso,@CGKFbandperformanceGray,0,0,num_stop,0,num_stop_2,maxOptLowerBound_2,num_stop_UpperBound);

[info_bestclassifier_3,OptSwarm_3,minmax_3,num_stop_3,k_3]=PsoProcessforCGKFband(num_particle_2,ParticleSize,CGKFbandScope,...
                    3,@InitCGSwarm,@BaseStepPso,@CGKFbandperformanceGray,0,0,num_stop,0,num_stop_3,maxOptLowerBound_3,num_stop_UpperBound);

[info_bestclassifier_4,OptSwarm_4,minmax_4,num_stop_4,k_4]=PsoProcessforCGKFband(num_particle_2,ParticleSize,CGKFbandScope,...
                    4,@InitCGSwarm,@BaseStepPso,@CGKFbandperformanceGray,0,0,num_stop,0,num_stop_4,maxOptLowerBound_4,num_stop_UpperBound);

% toc;
%% Testing
cmd_1 = ['-c ',num2str(2^info_bestclassifier_1(1)),' -g ',num2str(2^info_bestclassifier_1(2)),' -b 1'];
cmd_2 = ['-c ',num2str(2^info_bestclassifier_2(1)),' -g ',num2str(2^info_bestclassifier_2(2)),' -b 1'];
cmd_3 = ['-c ',num2str(2^info_bestclassifier_3(1)),' -g ',num2str(2^info_bestclassifier_3(2)),' -b 1'];
cmd_4 = ['-c ',num2str(2^info_bestclassifier_4(1)),' -g ',num2str(2^info_bestclassifier_4(2)),' -b 1'];

acc_4mdls = [info_bestclassifier_1(4),...
             info_bestclassifier_2(4),...
             info_bestclassifier_3(4),...
             info_bestclassifier_4(4)];

acc_all = zeros(1,4);

acc_4classes = zeros(4,4);
fband_1 = info_bestclassifier_1(3);
fband_graycode_1 = dec2gray(floor(fband_1),fea_num);
fband_mdl_1 = false(1,fea_num);
fband_mdl_1(1,:) = flipud(logical(str2num(fband_graycode_1(:))));
fband_2 = info_bestclassifier_2(3);
fband_graycode_2 = dec2gray(floor(fband_2),fea_num);
fband_mdl_23 = false(1,fea_num);
fband_mdl_23(1,:) = flipud(logical(str2num(fband_graycode_2(:))));
fband_3 = info_bestclassifier_3(3);
fband_graycode_3 = dec2gray(floor(fband_3),fea_num);
fband_mdl_24 = false(1,fea_num);
fband_mdl_24(1,:) = flipud(logical(str2num(fband_graycode_3(:))));
fband_4 = info_bestclassifier_4(3);
fband_graycode_4 = dec2gray(floor(fband_4),fea_num);
fband_mdl_34 = false(1,fea_num);
fband_mdl_34(1,:) = flipud(logical(str2num(fband_graycode_4(:))));
save('fband_selection.mat','fband_mdl_1','fband_mdl_23','fband_mdl_24','fband_mdl_34');
for valcount = 1:3
    load(['FeaSets_',num2str(valcount),'_for_classifiers_trte.mat']);

    model_1 = svmtrain(train_Fea_Y_1,train_Fea_1(:,fband_mdl_1),cmd_1);
    model_23 = svmtrain(train_Fea_Y_23,train_Fea_23(:,fband_mdl_23),cmd_2);
    model_24 = svmtrain(train_Fea_Y_24,train_Fea_24(:,fband_mdl_24),cmd_3);
    model_34 = svmtrain(train_Fea_Y_34,train_Fea_34(:,fband_mdl_34),cmd_4);
    svmModels = struct('Mdl_1', model_1, 'Mdl_23', model_23, 'Mdl_24', model_24, 'Mdl_34', model_34,...
            'cmd_1', cmd_1,'cmd_2', cmd_2,'cmd_3', cmd_3,'cmd_4', cmd_4);
    predict_label_te = svmPredict_4Class_ov3ovo_selectfband(test_Fea_all,test_Fea_Y,svmModels);

    acc_all(valcount) = sum(predict_label_te == test_Fea_Y)/size(test_Fea_Y,1);
    acc_4classes(1,valcount) = sum(predict_label_te(test_Fea_Y==1,1) == 1)/sum(test_Fea_Y == 1);
    acc_4classes(2,valcount) = sum(predict_label_te(test_Fea_Y==2,1) == 2)/sum(test_Fea_Y == 2);
    acc_4classes(3,valcount) = sum(predict_label_te(test_Fea_Y==3,1) == 3)/sum(test_Fea_Y == 3);
    acc_4classes(4,valcount) = sum(predict_label_te(test_Fea_Y==4,1) == 4)/sum(test_Fea_Y == 4);    
end

acc_all(4) = mean(acc_all(1:3));
acc_4classes(:,4) = mean(acc_4classes(:,1:3),2);

fprintf('      acc_all      = %2.4f %2.4f %2.4f   average %.4f\n',acc_all);
fprintf('      acc_4classes = %2.4f %2.4f %2.4f %2.4f\n',acc_4classes(:,4));
fprintf('      acc_4mdls    = %2.4f %2.4f %2.4f %2.4f\n',acc_4mdls);
toc;
save(FunctionNowFilename('fband_selection', '.mat'),...
     'fband_mdl_1','fband_mdl_23','fband_mdl_24','fband_mdl_34',...
     'acc_all','acc_4classes','acc_4mdls');
save(FunctionNowFilename('fband_selection_all', '.mat'));

pause;
%% generating initial models, using all samples
tic;
cmds = cell(5,1);
cmds{1} = svmModels.cmd_1;
cmds{2} = svmModels.cmd_2; 
cmds{3} = svmModels.cmd_3;
cmds{4} = svmModels.cmd_4;

[CSPs, svmModels] = model4allsample(FilterSample_Idle,FilterSample_Walk,FilterSample_Ascend,FilterSample_Descend,cmds,number_bandpass_filters,FilterNum);

save('ov3ovo_svm_models.mat', 'svmModels');
save('ov3ovo_CspTranspose_forUse.mat','CSPs','FilterNum');

% best_acc_te = acc_te(acc_te == acctmp);
% best_acc_4classes = acc_4classes(acc_te == acctmp,:);
% best_acc_4mdls = acc_4mdls(acc_te == acctmp,:);
% save('ov3ovo_Outcome','acc_te','acc_4classes','acc_4mdls');
save(FunctionNowFilename('InitialModel', '.mat'),...
    'svmModels',...
    'CSPs','FilterNum',...
    'acc_all','acc_4classes','acc_4mdls');

toc;
fprintf('Initial Model Generated!\n');

