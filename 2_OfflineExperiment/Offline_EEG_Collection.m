% This Script is for MI-BCI Discrete Trial-based offline training session.
% No VR feedback.
% Contains 4 types of motor imagery tasks: idle(Trigger=1),walk(Trigger=2),
% go upstairs(Trigger=3),go downstairs(Trigger=4).
% (For resting & foucusing stage of each trial, Trigger = 6)
% 4 task types * 30 trials/task type = 120 trials, where the task of each
% trials are randomly generated.

% 120 trials
% 30 trials/task type
% 4 sec per trial
% 2  valid sample per trial

% 第三秒电刺激同时显示人物画面


addpath(genpath('toolbox'));
pnet('closeall');
clc;
clear;
close all;
%% 启动FES程序，并初始化
% 程序说明：发送命令共6字节
%           Byte1: 0 start 100 stop
%           Byte2: current amplitude
%           Byte3: t_up
%           Byte4: t_flat
%           Byte5: t_down
%           Byte6: 1 left calf 2 left thigh 3 right thigh
system('E:\20220831_lty\fes\x64\Debug\fes.exe&'); 
pause(1);
StimControl = tcpip('localhost', 8888, 'NetworkRole', 'client','Timeout',1000);
StimControl.InputBuffersize = 1000;
StimControl.OutputBuffersize = 1000;
fopen(StimControl);
% StimCommand_0 = uint8(zeros(1,6)); !!!!! NEVERRR!
tStim = [3,14,2]; % [t_up,t_flat,t_down] * 100ms
StimCommand_1 = uint8([0,11,tStim,1]); % left calf
StimCommand_2 = uint8([0,7,tStim,2]); % left thigh
StimCommand_3 = uint8([0,9,tStim,3]); % right thigh 
% fwrite(StimControl,StimCommand_0);
% pause(2);

%% 设置脑电采集参数
init = 0;
freq = 256;
startStop = 1;
con = pnet('tcpconnect','127.0.0.1',4455);                                 % 建立一个连接
status = CheckNetStreamingVersion(con);                                    % 判断版本信息，正确返回状态值为1
[~, basicInfo] = ClientGetBasicMessage(con);                               % 获取设备基本信息basicInfo包含 size,eegChan,sampleRate,dataSize
[~, infoList] = ClientGetChannelMessage(con,basicInfo.eegChan);            % 获取通道信息

%% 启动Unity程序，并初始化
% 程序说明：发送命令共5字节
%           Byte1：画面/动作切换
%           Byte2：控制画面是否运动
%           Byte3：画面文字显示（离线训练实验无文字提示）
%           Byte4：动作类型
%           Byte5：预留

% system('C:\Users\win10\Desktop\UnityOnline0803\ClimbStair3.exe&');      % Unity动画exe文件地址
system('E:\20220831_lty\UnityOnline0908speedup\ClimbStair3.exe&'); 
pause(3)
UnityControl = tcpip('localhost', 8881, 'NetworkRole', 'client');          % 新的端口改为8881
fopen(UnityControl);
pause(1)
sendbuf = uint8(1:5);
sendbuf(1,1) = hex2dec('01') ;
sendbuf(1,2) = hex2dec('00') ;
sendbuf(1,3) = hex2dec('00') ;
sendbuf(1,4) = hex2dec('00') ;
sendbuf(1,5) = hex2dec('00') ;
fwrite(UnityControl,sendbuf);
pause(3)
%% 随机生成指定数量的trial
TrialNum = 80;                                                             % 设置采集trial数量
TrialIndex = randperm(TrialNum);
All_data = [];
Trigger = 0;
AllTrial = 0;
index1 = ones(TrialNum/4,1);                                               %size 15*1, all 1s，空想任务
index2 = ones(TrialNum/4,1)*2;                                             %size 15*1, all 2s，行走任务
index3 = ones(TrialNum/4,1)*3;                                             %size 15*1, all 3s，上台阶任务
index4 = ones(TrialNum/4,1)*4;                                             %size 15*1, all 4s，下台阶任务

randomindex = [index1;index2;index3;index4];                               %size 60*1
RandomTrial = randomindex(TrialIndex);                                     %size 60*1, RandomTrial=本次实验随机trial顺序

% RandomTrial(1:4)=[1,2,3,4];
%% 开始采集
Timer = 0;
TrialData = [];
data = [];
while (AllTrial <= TrialNum)
    if Timer==0  %提示专注
        Trigger = 6;
        sendbuf(1,1) = hex2dec('03') ;
        sendbuf(1,2) = hex2dec('00') ;
        sendbuf(1,3) = hex2dec('00') ;
        sendbuf(1,4) = hex2dec('00') ;
        fwrite(UnityControl,sendbuf);       
        AllTrial = AllTrial + 1;
    end
    if Timer==2  %电刺激并MI
        switch RandomTrial(AllTrial)
            case 2 % Walking task, left calf
                fwrite(StimControl,StimCommand_1);   
                fprintf('FES: left calf.\n');
            case 3 % Ascending task, left thigh
                fwrite(StimControl,StimCommand_2);  
                fprintf('FES: left thigh.\n');
            case 4 % Descending task, right thigh
                fwrite(StimControl,StimCommand_3);  
                fprintf('FES: right thigh.\n');
        end
        if RandomTrial(AllTrial)==1  % 空想任务
            sendbuf(1,1) = hex2dec('01') ;
            sendbuf(1,2) = hex2dec('00') ;
            sendbuf(1,3) = hex2dec('00') ;
            sendbuf(1,4) = hex2dec('00') ;
            fwrite(UnityControl,sendbuf);  
        end
        if RandomTrial(AllTrial)==2  % 步行任务
            sendbuf(1,1) = hex2dec('02') ;
            sendbuf(1,2) = hex2dec('01') ;
            sendbuf(1,3) = hex2dec('00') ;
            sendbuf(1,4) = hex2dec('01') ;
            fwrite(UnityControl,sendbuf);  
        end
        if RandomTrial(AllTrial)==3  % 上台阶任务
            sendbuf(1,1) = hex2dec('02') ;
            sendbuf(1,2) = hex2dec('01') ;
            sendbuf(1,3) = hex2dec('00') ;
            sendbuf(1,4) = hex2dec('02') ;
            fwrite(UnityControl,sendbuf);  
        end
        if RandomTrial(AllTrial)==4  % 下台阶任务
            sendbuf(1,1) = hex2dec('02') ;
            sendbuf(1,2) = hex2dec('01') ;
            sendbuf(1,3) = hex2dec('00') ;
            sendbuf(1,4) = hex2dec('03') ;
            fwrite(UnityControl,sendbuf);  
        end
    end
    if Timer==5  %开始打标签
        if RandomTrial(AllTrial)==1  % 空想任务
            Trigger = 1;
        end
        if RandomTrial(AllTrial)==2  % 步行任务
            Trigger = 2;
        end
        if RandomTrial(AllTrial)==3  % 上台阶任务
            Trigger = 3;
        end
        if RandomTrial(AllTrial)==4  % 下台阶任务
            Trigger = 4;
        end
    end
    if Timer==9  %开始休息
        Trigger = 6;
        sendbuf(1,1) = hex2dec('04') ;
        sendbuf(1,2) = hex2dec('00') ;
        sendbuf(1,3) = hex2dec('00') ;
        sendbuf(1,4) = hex2dec('00') ;
        fwrite(UnityControl,sendbuf);  
    end

    tic
    fprintf('AllTrial = %d, Timer = %d.\n',AllTrial,Timer);
    [~, data] = ClientGetDataPacket(con,basicInfo,infoList,startStop,init); % Obtain EEG data, 需要在ClientGetDataPacket设置要不要移除基线
% 	pause(1);
    fprintf('获取 1 秒脑电数据：');
    toc
    
    TriggerRepeat = repmat(Trigger,1,256);  
    data = [data;TriggerRepeat];
    TrialData = [TrialData,data];
    Timer = Timer + 1;
     
    if Timer == 11 % Start the next trial
        Timer = 0;
    end
    
end
%% 存储数据
system('taskkill /F /IM ClimbStair3.exe');
system('taskkill /F /IM fes.exe');
close all;
TrialData = TrialData(2:end,:);  %去掉矩阵第一行
ChanLabel = flip({infoList.chanLabel});
pnet('closeall');   % 将连接关闭

save(FunctionNowFilename('Offline_EEGdata_', '.mat' ),'TrialData','TrialIndex','ChanLabel');
