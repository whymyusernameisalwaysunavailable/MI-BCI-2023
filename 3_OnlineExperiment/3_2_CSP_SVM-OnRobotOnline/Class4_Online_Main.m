% This Script is for MI-BCI online training session (2 class).
% Contains 3 types of motor imagery tasks: walk(Tag=2),
% go upstairs(Tag=3),go downstairs(Tag=4).
% Idle(Tag=1) is used as initial screen.
% 3 Motor Imagination types， 60 trials, where the task of each
% trial is randomly generated.

%% Update
% rest and train model every 100 failures in a trial
% only 3 samples after fes used to update sample
% add unity text for 1.rest every 

%% Log
% Updated Time:     2022/10/14
% Author:           Chutian Zhang(张楚天); Shixin Ren(任士鑫)
% Version:          V 1.0.0
% File:             Online_Main.mat
% MATLAB Version:   R2018b
% Describe:         Write during the PhD work at CASIA, Github link: https://github.com/JodyZZZ/MI-BCI

%% Dependent ToolBox：
% Curry 8 software communication toolbox
% tcp_udp_ip_2.0.6

%% Clear all
addpath(genpath('toolbox'));
pnet('closeall');fclose all;clc;clear;close all;
%parpool(4);
isModelInitialized = 0;
while ~isModelInitialized
    isModelInitialized = input('Is Database & Model Initialized ? 1 = yes; 0 = no: ');
end

diary(FunctionNowFilename('outputlog_Main', '.txt' ));diary on;

%% Initialize EEG data accq connection
init = 0;
startStop = 1;
con = pnet('tcpconnect','127.0.0.1',4455);                                 % establish connection w/ Curry8 sw
status = CheckNetStreamingVersion(con);                                    % acquire version info, true=1
[~, basicInfo] = ClientGetBasicMessage(con);                               % acquire basicInfo: size,eegChan, sampleRate, dataSize
[~, infoList] = ClientGetChannelMessage(con,basicInfo.eegChan);            % acquire channel info

%% Launch and Initialize FES
% MatLab to FES Protocol:
%   Send:6-byte message:
%           Byte1: 0 start 100 stop
%           Byte2: current amplitude
%           Byte3: t_up
%           Byte4: t_flat
%           Byte5: t_down
%           Byte6: 1 left crus 2 left thigh 3 right thigh
system('E:\20220831_lty\fes\x64\Debug\fes.exe&');
pause(1);
StimControl = tcpip('localhost', 8888, 'NetworkRole', 'client','Timeout',1000);
StimControl.InputBuffersize = 1000;
StimControl.OutputBuffersize = 1000;
fopen(StimControl);
tStim = [3,14,2]; % [t_up,t_flat,t_down] * 100ms

StimCommand_1 = uint8([0,9,tStim,1]); % left calf
StimCommand_2 = uint8([0,7,tStim,2]); % left thigh
StimCommand_3 = uint8([0,7,tStim,3]); % right thigh 

%% Establish RobotControl comm.
global RobotControl
RobotControl=tcpip('localhost',5288,'NetworkRole','client');
fopen(RobotControl);

t_robotonset = [0,3.7,1.5,1];  % time for robot to get ready to move
t_robotoffset = [0,2,13,12];  % time for robot and unity to finish and stop moving
t_robotonset2 = [0,3.7,1.5,1];


%% Scene & trial info. initialization and saving
SceneNum = 12;
rng(1);
SceneIndex = randperm(SceneNum);

index2 = ones(SceneNum/3,1)*2;                                             %size 20*1, all ？s，行走任务
index3 = ones(SceneNum/3,1)*3;                                             %size 20*1, all ？s，上台阶任务
index4 = ones(SceneNum/3,1)*4;                                             %size 20*1, all ？s，下台阶任务

randomindex = [index2;index3;index4];                                      %size 60*1
RandomScene = randomindex(SceneIndex);                                     %size 60*1, RandomScene=本次实验随机trial顺序

RandomScene(1:4) = [2; 3; 2; 4];                                           %手动设置开头的四个场景
i = 1;
while i < SceneNum
    switch RandomScene(i)*RandomScene(i+1)
        case 4               
            RandomScene(i+1) = 3;
        case 9
            RandomScene(i+1) = 2;
        case 16
            RandomScene(i+1) = 2;
        case 12
            RandomScene(i+1) = 2;
    end
    i = i + 1;
end

RandomScene = [4;2;3;3;2;4;4;2;3;3;2;4];

SceneCount = 0;
TrialCount=0;
AllpredictLabel =cell(SceneNum*2,1);
updateindx_walk = 1;    % data cell updating index
updateindx_ascend = 1;
updateindx_descend = 1;

save('updateinfo.mat','SceneNum','SceneCount','TrialCount','RandomScene',...
                      'AllpredictLabel',...
                      'updateindx_walk','updateindx_ascend','updateindx_descend');


%% Establish MatLab_Main to MatLab_ModelUpdate comm.
global MatlabComm_Main
bytesToRead_ModelUpdate = 1;
MatlabComm_Main = tcpip('localhost',9991,'NetworkRole','server');          % set TCP/IP conn. server end, PortNum: 9991
MatlabComm_Main.InputBuffersize = 10000;
MatlabComm_Main.OutputBuffersize = 10000;
MatlabComm_Main.BytesAvailableFcn = {@Online_GetModelUpdateFeedback,bytesToRead_ModelUpdate};
MatlabComm_Main.BytesAvailableFcnMode = 'byte';
MatlabComm_Main.BytesAvailableFcnCount = bytesToRead_ModelUpdate;
fopen(MatlabComm_Main);                                                    % start the server, return until connection estabilished
pause(0.1);

% MatLan_Main to MatLab_ModelUpdate Protocol
%   Receive:1-byte message:
%       Byte 1: model file update status
global dataReceive_FromModelUpdate;
dataReceive_FromModelUpdate = [];

%   Send:1-byte message:
%       Byte 1: data file update status
dataSend_ToModelUpdate = uint8(0);
fwrite(MatlabComm_Main,dataSend_ToModelUpdate);
pause(0.1);
[~,ModelUpdateOutputSize] = size(dataReceive_FromModelUpdate);


%% Launch and initialize Unity program, establish Unity-MatLab_Main comm.
system('E:\20220831_lty\UnityOnline1015moretext\ClimbStair3.exe&');         % Unity animation .exe file address. PS.Use online .exe
pause(3);
global UnityControl;
UnityControl = tcpip('localhost', 8881, 'NetworkRole', 'client');          % set TCP/IP conn. client end,PortNum: 8881
bytesToRead_Unity = 1;
UnityControl.InputBufferSize = 1000;
UnityControl.BytesAvailableFcn = {@Online_GetUnityFeedback,bytesToRead_Unity};
UnityControl.BytesAvailableFcnMode = 'byte';
UnityControl.BytesAvailableFcnCount = bytesToRead_Unity;
fopen(UnityControl);
pause(1)

% MatLab to Unity Protocol:
%   Send:5-byte message:
%       Byte 1: Scene Switch
%       Byte 2: Scene & character in-motion control
%       Byte 3: Text display
%       Byte 4: Action to be generated
%       Byte 5: Reserve
dataSend_ToUnity = uint8(1:5);                                             % Buffer for control commands to Unity animation
dataSend_ToUnity(1,1) = hex2dec('01') ;
dataSend_ToUnity(1,2) = hex2dec('00') ;
dataSend_ToUnity(1,3) = hex2dec('00') ;
dataSend_ToUnity(1,4) = hex2dec('00') ;
dataSend_ToUnity(1,5) = hex2dec('00') ;
fwrite(UnityControl,dataSend_ToUnity);

% Unity to Matlab Protocol:
%   Receive:1-byte message:
%       Byte 1: Animation Action Completion Status
global dataReceive_FromUnity;
dataReceive_FromUnity = [];
[~,UnityOutputSize] = size(dataReceive_FromUnity);


%% Load baseline/training data
global TrialData;
% 
% trials_per_action = 20;                                                    % Trial nummber/action in offline task
% sample_frequency = 256;
% seconds_per_TrainTrial = 4;
% 
% WindowLength = 512;                                                        % Window length
% SlideWindowLength = 256;                                                   % Slide length
load('InitializationParameters.mat');

data_points_per_TrainTrial = sample_frequency * seconds_per_TrainTrial;
windows_per_trial = (data_points_per_TrainTrial - WindowLength) / SlideWindowLength + 1;
windows_per_action = windows_per_trial * trials_per_action;

load('SlideDataUpdate_Base.mat');
SlideDataUpdate_Idle = SlideDataUpdate_Idle_Base;
SlideDataUpdate_Walk = SlideDataUpdate_Walk_Base;
SlideDataUpdate_Ascend = SlideDataUpdate_Ascend_Base;
SlideDataUpdate_Descend = SlideDataUpdate_Descend_Base;

save('SlideDataUpdate_Idle.mat','SlideDataUpdate_Idle');
save('SlideDataUpdate_Walk.mat','SlideDataUpdate_Walk');
save('SlideDataUpdate_Ascend.mat','SlideDataUpdate_Ascend');
save('SlideDataUpdate_Descend.mat','SlideDataUpdate_Descend');
% 保存更新的样本集

sceneTag = 0;
% SlideSample_test = [];
% save('OnlineSlideSample_forTest.mat','SlideSample_test');

OnlineSlideData_all = [];
OnlineSlideData_valid = [];

global rawSlideData_withtag;
rawSlideData_withtag = [];
rawSlideData_valid = [];
maxupdatenum_SlideData = 20; % as named
trials_per_unit = 12;
maxMItime_fes = 6; % after which give fes again
maxMItime_sampleupdate = 21; % after which pause and update sample 
istoUpdate = 0;
time_modelupdate = 45;
time_interunit = 30;

%% Start test
Index = 0;

TrialData = [];                                                            % clear out TrialData
global PredictType;
PredictType_pre = 0;
predictcount_tmp = 0;   % prediction times until correct
predictcount_walk = [];
predictcount_ascend = [];
predictcount_descend = [];
predictLabel = [];

while TrialCount <= SceneNum*2
    
    if Index == 0     % to Unity animation，提示专注
        dataSend_ToUnity(1,1) = hex2dec('03') ;
        dataSend_ToUnity(1,2) = hex2dec('00') ;
        dataSend_ToUnity(1,3) = hex2dec('00') ;
        dataSend_ToUnity(1,4) = hex2dec('00') ;
        fwrite(UnityControl,dataSend_ToUnity);
        TrialCount = TrialCount + 1;
        SceneCount = floor((TrialCount+1)/2);
        pause(2);
        Index = 1;
    end
    
    if Index == 1     % send all 60 trials to Unity（随机任务，停止，“请想象动作”）
        SendIndex = 1;
        tic;
        fprintf('  生成场景...')
        FinishSendingFlag = Online_SendScenes(SendIndex,RandomScene,SceneNum);
        toc;
        pause(3);
        
        if FinishSendingFlag == 1                                          % Matlab have sent scenes
            datatmp_FromUnity = dataReceive_FromUnity;
            if size(datatmp_FromUnity,2) == SceneNum                   % Unity acked all scenes generated
                fprintf('%d random generated scenes have been displayed!\n\n',SceneNum);
            else
                fprintf('Unity fail to generate all scenes! (%d scenes have been displayed.)\n\n',size(datatmp_FromUnity,2))
                quit;
            end
        else
            disp('MatLab fail to send scenes!')
            quit;
        end
        
        fprintf('____________________________No.%d.Trial____________________________\n',TrialCount);
        fprintf('No.%d.Scene 场景类型：%d.\n',SceneCount,RandomScene(SceneCount));
        
        dataSend_ToUnity(1,1) = hex2dec('02') ;
        dataSend_ToUnity(1,2) = hex2dec('00') ;                    % start Unity animation's motion
        dataSend_ToUnity(1,3) = hex2dec('04');                     % text prompt“想象准备”
        dataSend_ToUnity(1,4) = hex2dec('00') ;
        fwrite(UnityControl,dataSend_ToUnity);
        fprintf('Unity:想象准备。\n');
        predictcount_tmp = 0;
        PredictType_pre = 0;
        sceneTag = RandomScene(SceneCount);
        
        tic;
        fprintf('        ');
        [~, data] = ClientGetDataPacket(con,basicInfo,infoList,startStop,init);
        switch RandomScene(SceneCount)
            case 2 % Walking task, left crus
                fwrite(StimControl,StimCommand_1);
                fprintf('FES: left crus.\n');
            case 3 % Ascending task, left thigh
                fwrite(StimControl,StimCommand_2);
                fprintf('FES: left thigh.\n');
            case 4 % Descending task, right thigh
                fwrite(StimControl,StimCommand_3);
                fprintf('FES: right thigh.\n');
        end
        fprintf('        ');
        [~, data] = ClientGetDataPacket(con,basicInfo,infoList,startStop,init);
        fprintf('        ');
        [~, data] = ClientGetDataPacket(con,basicInfo,infoList,startStop,init);
        
        dataSend_ToUnity(1,2) = hex2dec('00') ;                        % start Unity animation's motion
        dataSend_ToUnity(1,3) = hex2dec('01');                         % text prompt“请想象动作”
        dataSend_ToUnity(1,4) = hex2dec('00') ;
        fwrite(UnityControl,dataSend_ToUnity);
        fprintf('Unity:请想象动作。\n');
        
        fprintf('        ');
        [~, data] = ClientGetDataPacket(con,basicInfo,infoList,startStop,init);
        fprintf('	获取提示后 4 秒脑电数据：(1s预备+2s电刺激+1s空)');
        toc;
        tic;
        fprintf('        ');
        [~, data] = ClientGetDataPacket(con,basicInfo,infoList,startStop,init);     % acq EEG data(note: set don't remove baseline in ClientGetDataPacket)
        fprintf('	获取提示后第 5 秒脑电数据：');
        toc;
        %         if mean(mean(abs(data(channels+1,1:256)),2))>15
        %             fprintf('warning: 脑电信号异常,请重新开始实验！！');
        %             break;
        %         end
        tic; % record time interval between the two adjacent data aquisitions
        TagRepeat = repmat(sceneTag,1,size(data,2));                       % add label to the last row
        data = [data;TagRepeat];                                           % append label to acq.ed EEG data
        data = data(2:end,:);                                              % remove first row of acq.ed EEG data
        
        % !!!
        TrialData = [];
        % !!!
        TrialData = [TrialData,data];
        TrialData = double(TrialData);
        data = [];
        
        Index = 2;
        fprintf('	获取第 1 秒带标签数据样本（1/2样本）：');
        
    end
    
    if Index == 1.5 % start data aquisition
        dataSend_ToUnity(1,2) = hex2dec('00') ;                    % start Unity animation's motion
        dataSend_ToUnity(1,3) = hex2dec('04');                     % text prompt“想象准备”
        dataSend_ToUnity(1,4) = hex2dec('00') ;
        fwrite(UnityControl,dataSend_ToUnity);
        fprintf('Unity:想象准备。\n');

        t = 0;
        n = 0;
        while t < 0.3
            data = [];
            tic;
            fprintf('        ');
            [~, data] = ClientGetDataPacket(con,basicInfo,infoList,startStop,init);     % acq EEG data(note: set don't remove baseline in ClientGetDataPacket)
            t = toc;
            n = n + 1;
        end
        fprintf('	n = %d.\n',n);
        fprintf('	开始接收实时脑电数据。\n');

        tic;
        fprintf('        ');
        [~, data] = ClientGetDataPacket(con,basicInfo,infoList,startStop,init);     % acq EEG data(note: set don't remove baseline in ClientGetDataPacket)
        switch RandomScene(SceneCount)
            case 2 % Walking task, left crus
                fwrite(StimControl,StimCommand_1);
                fprintf('FES: left crus.\n');
            case 3 % Ascending task, left thigh
                fwrite(StimControl,StimCommand_2);
                fprintf('FES: left thigh.\n');
            case 4 % Descending task, right thigh
                fwrite(StimControl,StimCommand_3);
                fprintf('FES: right thigh.\n');
        end
        fprintf('        ');
        [~, data] = ClientGetDataPacket(con,basicInfo,infoList,startStop,init);     % acq EEG data(note: set don't remove baseline in ClientGetDataPacket)
        fprintf('        ');
        [~, data] = ClientGetDataPacket(con,basicInfo,infoList,startStop,init);     % acq EEG data(note: set don't remove baseline in ClientGetDataPacket)

        dataSend_ToUnity(1,2) = hex2dec('00') ;                        % start Unity animation's motion
        dataSend_ToUnity(1,3) = hex2dec('01');                         % text prompt“请想象动作”
        dataSend_ToUnity(1,4) = hex2dec('00') ;
        fwrite(UnityControl,dataSend_ToUnity);
        fprintf('Unity:请想象动作。\n');

        fprintf('        ');
        [~, data] = ClientGetDataPacket(con,basicInfo,infoList,startStop,init);     % acq EEG data(note: set don't remove baseline in ClientGetDataPacket)
        fprintf('	获取提示后 4 秒脑电数据：(1s预备+2s电刺激+1s空)');
        toc;
        tic;
        data = [];
        fprintf('        ');
        [~, data] = ClientGetDataPacket(con,basicInfo,infoList,startStop,init);     % acq EEG data(note: set don't remove baseline in ClientGetDataPacket)
        fprintf('	获取提示后第 5 秒脑电数据：');
        toc;
        tic;
        TagRepeat = repmat(sceneTag,1,size(data,2));
        data = [data;TagRepeat];
        data = data(2:end,:);
        TrialData = [TrialData,data];
        TrialData = double(TrialData);
        data = [];                                                     % clear out data matrix          
        Index = 2;                                                 % continue acq EEG data --> predict
        continue;        
    end
    
    if Index == 2    % Acquire EEG data, save and add label to the last row (TagRepeat)
        toc;
        if mod(predictcount_tmp+1,maxMItime_fes)==1 && predictcount_tmp~=0
            tic;
            fprintf('\n超过最大想象时间，重新电刺激.\n');
            switch RandomScene(SceneCount)
                case 2 % Walking task, left crus
                    fwrite(StimControl,StimCommand_1);
                    fprintf('FES: left crus.\n');
                case 3 % Ascending task, left thigh
                    fwrite(StimControl,StimCommand_2);
                    fprintf('FES: left thigh.\n');
                case 4 % Descending task, right thigh
                    fwrite(StimControl,StimCommand_3);
                    fprintf('FES: right thigh.\n');
            end
            fprintf('        ');
            [~, data] = ClientGetDataPacket(con,basicInfo,infoList,startStop,init);
            fprintf('        ');
            [~, data] = ClientGetDataPacket(con,basicInfo,infoList,startStop,init);
            fprintf('        ');
            [~, data] = ClientGetDataPacket(con,basicInfo,infoList,startStop,init);
            fprintf('	获取电刺激后 3 秒脑电数据：');
            toc;
            tic;
            fprintf('        ');
            [~, data] = ClientGetDataPacket(con,basicInfo,infoList,startStop,init);
            fprintf('	获取 1 秒脑电数据:');
            toc;
            TagRepeat = repmat(sceneTag,1,size(data,2));                    % add label to the last row
            data = [data;TagRepeat];                                       % append label to acq.ed EEG data
            data = data(2:end,:);                                              % remove first row of acq.ed EEG data
            TrialData = [TrialData,data];
            TrialData = double(TrialData);
            data = [];                                                         % clear out data matrix
        end
        
        tic;
        fprintf('        ');
        [~, data] = ClientGetDataPacket(con,basicInfo,infoList,startStop,init);     % acq EEG data(note: set don't remove baseline in ClientGetDataPacket)
        fprintf('	获取 1 秒脑电数据：');
        toc;
        tic;
        TagRepeat = repmat(sceneTag,1,size(data,2));                       % add label to the last row
        data = [data;TagRepeat];                                           % append label to acq.ed EEG data
        data = data(2:end,:);                                              % remove first row of acq.ed EEG data
        TrialData = [TrialData,data];
        TrialData = double(TrialData);
        size_Y = size(TrialData,2);
        rawSlideData_withtag = TrialData(:,(size_Y - WindowLength + 1):size_Y);
        data = [];                                                         % clear out data matrix
        
        Index = 3;
    end
        
    if Index == 3 % predict, update, execute
        % figure out model update status through MatLab_Main to MatLab_ModelUpdate comm.
        for m=1:1000
            dataTmp_FromModelUpdate = dataReceive_FromModelUpdate;
            if dataTmp_FromModelUpdate(end) == 1
                disp('Model updating......');
                pause(1);
                continue;
            else
                predictcount_tmp = predictcount_tmp + 1;
                fprintf('第 %d 次预测：',predictcount_tmp);
                
                if dataTmp_FromModelUpdate(end) == 2
                    disp('	Using updated model......');
                elseif dataTmp_FromModelUpdate(end) == 0
                    disp('	Using original model......');
                end
                
                Class4_Online_Predict_ov3ovo;
                
                OnlineSlideData_all = [OnlineSlideData_all,{rawSlideData_withtag}];
                if PredictType ~= 0 && PredictType ~= 1 && mod(predictcount_tmp,maxMItime_fes)<=3
                    rawSlideData_valid = [rawSlideData_valid,{rawSlideData_withtag}];
                end
                rawSlideData_withtag = [];
                break;
            end
        end
        
        % Cases that dataset should be updated
        if mod(predictcount_tmp,maxMItime_sampleupdate)== 0 || PredictType == RandomScene(SceneCount)
            istoUpdate = 1;
        else
            istoUpdate = 0;
        end
        
        predictLabel = [predictLabel;PredictType];
        if istoUpdate == 1                                                 % if prediction=current label, MI success, !需试对不对
            [~,numSlideWindow_valid] = size(rawSlideData_valid); % should be same as 
%             numSlideWindow_valid = sum(predictLabel ~= 1 & predictLabel ~= 0);     
            numupdate = min(numSlideWindow_valid,maxupdatenum_SlideData);
            
            if PredictType == RandomScene(SceneCount)
                AllpredictLabel{TrialCount,1}= predictLabel;
                predictLabel= [];
                fprintf('	预测结果为：%d，预测成功！！！\n\n',PredictType);
                dataSend_ToUnity(1,2) = hex2dec('00') ;                        % start Unity animation's motion
                dataSend_ToUnity(1,3) = hex2dec('02');                         % text prompt“想象执行中”
                dataSend_ToUnity(1,4) = hex2dec('00') ;
                fwrite(UnityControl,dataSend_ToUnity);
                fprintf('Unity:想象执行中。\n');
            end

            % Unity Animation execute, replace successful 2-sec MI to dataset from cell 1
            if RandomScene(SceneCount)==2
                disp('更新步行样本...');
                if PredictType == RandomScene(SceneCount)
                    predictcount_walk = [predictcount_walk, predictcount_tmp];
                    fprintf('	步行平均需要预测次数为 %d 次。\n',mean(predictcount_walk));
                end
                
                newupdateindx_walk = mod((updateindx_walk + numupdate),windows_per_action);
                if updateindx_walk <= (windows_per_action - numupdate + 1)
                    SlideDataUpdate_Walk(updateindx_walk:(newupdateindx_walk-1)) = rawSlideData_valid((numSlideWindow_valid-numupdate+1):numSlideWindow_valid);
                    % if numupdate < numSlideWindow_valid then pick last
                    % numupdate of rawSlideData to update
                    % to be check !!!!!!!!!!!!!
                else
                    SlideDataUpdate_Walk(updateindx_walk:windows_per_action) = rawSlideData_valid((numSlideWindow_valid-numupdate+1):(numSlideWindow_valid-numupdate+windows_per_action-updateindx_walk+1));
                    SlideDataUpdate_Walk(1:(newupdateindx_walk-1)) = rawSlideData_valid((numSlideWindow_valid-numupdate+windows_per_action-updateindx_walk+2):(numSlideWindow_valid-numupdate+numupdate));
                end
                
                dataSend_ToModelUpdate(1) = hex2dec('00') ;            % Notify ModelUpdate dataset is updating
                fwrite(MatlabComm_Main,dataSend_ToModelUpdate);

                save('SlideDataUpdate_Walk.mat','SlideDataUpdate_Walk');
                save('updateinfo.mat','SceneNum','SceneCount','TrialCount','RandomScene',...
                                      'AllpredictLabel',...
                                      'updateindx_walk','updateindx_ascend','updateindx_descend','numupdate',...
                                      'predictcount_walk','predictcount_ascend','predictcount_descend');
                                  
                updateindx_walk = newupdateindx_walk;
            end
            
            if RandomScene(SceneCount)==3
                disp('更新上台阶样本...');
                
                if PredictType == RandomScene(SceneCount)
                    predictcount_ascend = [predictcount_ascend, predictcount_tmp];
                    fprintf('	上台阶平均需要预测次数为 %d 次。\n',mean(predictcount_ascend));
                end
                
                newupdateindx_ascend = mod((updateindx_ascend + numupdate),windows_per_action);
                if updateindx_ascend <= (windows_per_action - numupdate + 1)
                    SlideDataUpdate_Ascend(updateindx_ascend:(newupdateindx_ascend-1)) = rawSlideData_valid((numSlideWindow_valid-numupdate+1):numSlideWindow_valid);
                    % if numupdate < numSlideWindow_valid then pick last
                    % numupdate of rawSlideData to update
                    % to be check !!!!!!!!!!!!!
                else
                    SlideDataUpdate_Ascend(updateindx_ascend:windows_per_action) = rawSlideData_valid((numSlideWindow_valid-numupdate+1):(numSlideWindow_valid-numupdate+windows_per_action-updateindx_ascend+1));
                    SlideDataUpdate_Ascend(1:(newupdateindx_ascend-1)) = rawSlideData_valid((numSlideWindow_valid-numupdate+windows_per_action-updateindx_ascend+2):(numSlideWindow_valid-numupdate+numupdate));
                end
                
                dataSend_ToModelUpdate(1) = hex2dec('00') ;            % Notify ModelUpdate dataset is updating
                fwrite(MatlabComm_Main,dataSend_ToModelUpdate);

                save('SlideDataUpdate_Ascend.mat','SlideDataUpdate_Ascend');
                save('updateinfo.mat','SceneNum','SceneCount','TrialCount','RandomScene',...
                                      'AllpredictLabel',...
                                      'updateindx_walk','updateindx_ascend','updateindx_descend','numupdate',...
                                      'predictcount_walk','predictcount_ascend','predictcount_descend');
                                  
                updateindx_ascend = newupdateindx_ascend;
            end
            
            if RandomScene(SceneCount)==4
                disp('更新下台阶样本...');
                if PredictType == RandomScene(SceneCount)
                    predictcount_descend = [predictcount_descend, predictcount_tmp];
                    fprintf('	下台阶平均需要预测次数为 %d 次。\n',mean(predictcount_descend));
                end
                
                newupdateindx_descend = mod((updateindx_descend + numupdate),windows_per_action);
                if updateindx_descend <= (windows_per_action - numupdate + 1)
                    SlideDataUpdate_Descend(updateindx_descend:(newupdateindx_descend-1)) = rawSlideData_valid((numSlideWindow_valid-numupdate+1):numSlideWindow_valid);
                    % if numupdate < numSlideWindow_valid then pick last
                    % numupdate of rawSlideData to update
                    % to be check !!!!!!!!!!!!!
                else
                    SlideDataUpdate_Descend(updateindx_descend:windows_per_action) = rawSlideData_valid((numSlideWindow_valid-numupdate+1):(numSlideWindow_valid-numupdate+windows_per_action-updateindx_descend+1));
                    SlideDataUpdate_Descend(1:(newupdateindx_descend-1)) = rawSlideData_valid((numSlideWindow_valid-numupdate+windows_per_action-updateindx_descend+2):(numSlideWindow_valid-numupdate+numupdate));
                end
                
                dataSend_ToModelUpdate(1) = hex2dec('00') ;            % Notify ModelUpdate dataset is updating
                fwrite(MatlabComm_Main,dataSend_ToModelUpdate);

                save('SlideDataUpdate_descend.mat','SlideDataUpdate_Descend');
                save('updateinfo.mat','SceneNum','SceneCount','TrialCount','RandomScene',...
                                      'AllpredictLabel',...
                                      'updateindx_walk','updateindx_ascend','updateindx_descend','numupdate',...
                                      'predictcount_walk','predictcount_ascend','predictcount_descend');
                                  
                updateindx_descend = newupdateindx_descend;
            end
            
            dataSend_ToModelUpdate(1) = TrialCount;
            fwrite(MatlabComm_Main,dataSend_ToModelUpdate);
            OnlineSlideData_valid = [OnlineSlideData_valid,rawSlideData_valid];
            rawSlideData_valid = [];  
            if PredictType == RandomScene(SceneCount)
                switch RandomScene(SceneCount)
                    case 2
                        textSend='Y1';
                        pause(0.1);
                        fwrite(RobotControl,textSend);
                    case 3
                        textSend='Y2';
                        pause(0.1);
                        fwrite(RobotControl,textSend);
                    case 4
                        textSend='Y3';
                        pause(0.1);
                        fwrite(RobotControl,textSend);
                end

                fprintf('Robot:发出运动指令。');
                toc;
                fprintf('Robot:等待进入初始状态。');
                pause(t_robotonset(RandomScene(SceneCount)));  % wait for robot to get ready to move
                toc;
                
                predictcount_tmp = 0;
                datatmp_FromUnity = dataReceive_FromUnity;
                while datatmp_FromUnity(end) ~= 2 && sum(datatmp_FromUnity(end-1:end) == [2,4]) ~= 2 % 有可能收到2之后马上收到4，此时会再发一次，因此动画超前一个trial
                    dataSend_ToUnity(1,2) = hex2dec('01') ;
                    dataSend_ToUnity(1,3) = hex2dec('00') ;
                    dataSend_ToUnity(1,4) = hex2dec('00') ;
                    dataSend_ToUnity(1,5) = hex2dec('01') ;                
                    fwrite(UnityControl,dataSend_ToUnity);                 % Unity animation continue 运动，无文字
                    pause(0.1);
                    datatmp_FromUnity = dataReceive_FromUnity;
                end
                disp('Unity:人物开始运动。');
                
                switch RandomScene(SceneCount)
                    case 2 % Walking task, left crus
                        fwrite(StimControl,StimCommand_1);
                        fprintf('FES: left crus.\n');
                    case 3 % Ascending task, left thigh
                        fwrite(StimControl,StimCommand_2);
                        fprintf('FES: left thigh.\n');
                    case 4 % Descending task, right thigh
                        fwrite(StimControl,StimCommand_3);
                        fprintf('FES: right thigh.\n');
                end
                disp('......');
                Index = 4;
                continue                
            else
                fprintf('	预测结果为：%d， 与场景类别不符，预测失败！\n',PredictType);
                fprintf('	第 %d 次预测失败，暂停并更新模型。\n',predictcount_tmp);
                PredictType_pre = PredictType;
                PredictType = 0;                                           % Reset parameter
                
                dataSend_ToUnity(1,2) = hex2dec('00') ;                    % start Unity animation's motion
                dataSend_ToUnity(1,3) = hex2dec('06');                     % text prompt“在线样本数已达100个，请休息片刻”
                dataSend_ToUnity(1,4) = hex2dec('00') ;
                fwrite(UnityControl,dataSend_ToUnity);
                fprintf('Unity:在线样本数已达 %d 个，请休息片刻。\n\n',maxMItime_sampleupdate);
                pause(time_modelupdate-3);
                
                Index = 1.5;
                continue;
            end            
        else                                                               % if prediction/=current label, MI failed
            fprintf('	预测结果为：%d， 与场景类别不符，预测失败！',PredictType);
            PredictType_pre = PredictType;
            PredictType = 0;
            Index = 2;                                                     % cont. acq EEG data and predict
            continue
        end
    end
    
    if Index == 4 % waiting for 3rd-stride signal from Unity, similar to index = 1
        datatmp_FromUnity = dataReceive_FromUnity;
        
        if datatmp_FromUnity(end) == 3 % finished 3rd stride
            fprintf('No. %d. trial finished 1.5 strides!\n',TrialCount);
            if mod(TrialCount+1,trials_per_unit) ~= 1 
                dataSend_ToUnity(1,3) = hex2dec('03');                     % text prompt“本动作即将完成，请想象下一动作”
                dataSend_ToUnity(1,5) = hex2dec('00');
                fwrite(UnityControl,dataSend_ToUnity);
                fprintf('Move on to the next trial''s MI prediction.\n\n');
            else
                dataSend_ToUnity(1,3) = hex2dec('05');                     % text prompt“已完成一个单元，请休息一分钟”
                dataSend_ToUnity(1,5) = hex2dec('00');
                fwrite(UnityControl,dataSend_ToUnity);
                fprintf('\nUnity:已完成一个单元，请休息一分钟。\n');
            end
            
            TrialCount = TrialCount + 1;                                   % Move on to the next action's MI prediction
            SceneCount = floor((TrialCount+1)/2);
            if SceneCount > SceneNum
                break;
            end
            
            fprintf('____________________________No.%d.Trial____________________________\n',TrialCount);
            fprintf('No.%d.Scene 场景类型：%d.\n',SceneCount,RandomScene(SceneCount));
            datatmp_FromUnity(end) = 0;
            sceneTag = RandomScene(SceneCount);                            % set Tag value according to current MI type
            
            PredictType_pre = 0;
            fprintf(' ');
            toc;
            fprintf('Robot & Unity:等待停止。');
            pause(t_robotoffset(RandomScene(floor(TrialCount/2))));        % wait for robot & unity to finish and stop moving
            fprintf('	距离上次获取脑电数据：');
            toc;
            
            if mod(TrialCount,trials_per_unit) == 1 && TrialCount ~=1
                dataSend_ToUnity(1,2) = hex2dec('00') ;                    % stop Unity animation's motion
                dataSend_ToUnity(1,4) = hex2dec('00') ;
                fwrite(UnityControl,dataSend_ToUnity);
                fprintf('暂停 %d 秒。\n',time_interunit);
                pause(time_interunit);
            end
            
            Index = 1.5;
            continue
        else
            pause(0.1);
            Index = 4;
            continue
        end
    end
    
end  

% End the session when all trials are completed
if SceneCount == SceneNum+1
    
    disp('________________________________END________________________________');
    disp('This session has ended! Thank you!');
    
end

%% finish and save all acquired EEG data
close all;
pnet('closeall')  ;                                                         % Close pnet connection
fclose(UnityControl);                                                      % Close TCP/IP connections
fclose(MatlabComm_Main);
system('taskkill /F /IM ClimbStair3.exe');
system('taskkill /F /IM fes.exe');

% ChanLabel = flip({infoList.chanLabel});
save(FunctionNowFilename('ooooout', '.mat' ),'AllpredictLabel');  % save all online test EEG data

diary off;
%delete(gcp('nocreate'));