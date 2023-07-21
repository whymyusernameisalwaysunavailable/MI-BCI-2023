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

% �������̼�ͬʱ��ʾ���ﻭ��


addpath(genpath('toolbox'));
pnet('closeall');
clc;
clear;
close all;
%% ����FES���򣬲���ʼ��
% ����˵�����������6�ֽ�
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

%% �����Ե�ɼ�����
init = 0;
freq = 256;
startStop = 1;
con = pnet('tcpconnect','127.0.0.1',4455);                                 % ����һ������
status = CheckNetStreamingVersion(con);                                    % �жϰ汾��Ϣ����ȷ����״ֵ̬Ϊ1
[~, basicInfo] = ClientGetBasicMessage(con);                               % ��ȡ�豸������ϢbasicInfo���� size,eegChan,sampleRate,dataSize
[~, infoList] = ClientGetChannelMessage(con,basicInfo.eegChan);            % ��ȡͨ����Ϣ

%% ����Unity���򣬲���ʼ��
% ����˵�����������5�ֽ�
%           Byte1������/�����л�
%           Byte2�����ƻ����Ƿ��˶�
%           Byte3������������ʾ������ѵ��ʵ����������ʾ��
%           Byte4����������
%           Byte5��Ԥ��

% system('C:\Users\win10\Desktop\UnityOnline0803\ClimbStair3.exe&');      % Unity����exe�ļ���ַ
system('E:\20220831_lty\UnityOnline0908speedup\ClimbStair3.exe&'); 
pause(3)
UnityControl = tcpip('localhost', 8881, 'NetworkRole', 'client');          % �µĶ˿ڸ�Ϊ8881
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
%% �������ָ��������trial
TrialNum = 80;                                                             % ���òɼ�trial����
TrialIndex = randperm(TrialNum);
All_data = [];
Trigger = 0;
AllTrial = 0;
index1 = ones(TrialNum/4,1);                                               %size 15*1, all 1s����������
index2 = ones(TrialNum/4,1)*2;                                             %size 15*1, all 2s����������
index3 = ones(TrialNum/4,1)*3;                                             %size 15*1, all 3s����̨������
index4 = ones(TrialNum/4,1)*4;                                             %size 15*1, all 4s����̨������

randomindex = [index1;index2;index3;index4];                               %size 60*1
RandomTrial = randomindex(TrialIndex);                                     %size 60*1, RandomTrial=����ʵ�����trial˳��

% RandomTrial(1:4)=[1,2,3,4];
%% ��ʼ�ɼ�
Timer = 0;
TrialData = [];
data = [];
while (AllTrial <= TrialNum)
    if Timer==0  %��ʾרע
        Trigger = 6;
        sendbuf(1,1) = hex2dec('03') ;
        sendbuf(1,2) = hex2dec('00') ;
        sendbuf(1,3) = hex2dec('00') ;
        sendbuf(1,4) = hex2dec('00') ;
        fwrite(UnityControl,sendbuf);       
        AllTrial = AllTrial + 1;
    end
    if Timer==2  %��̼���MI
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
        if RandomTrial(AllTrial)==1  % ��������
            sendbuf(1,1) = hex2dec('01') ;
            sendbuf(1,2) = hex2dec('00') ;
            sendbuf(1,3) = hex2dec('00') ;
            sendbuf(1,4) = hex2dec('00') ;
            fwrite(UnityControl,sendbuf);  
        end
        if RandomTrial(AllTrial)==2  % ��������
            sendbuf(1,1) = hex2dec('02') ;
            sendbuf(1,2) = hex2dec('01') ;
            sendbuf(1,3) = hex2dec('00') ;
            sendbuf(1,4) = hex2dec('01') ;
            fwrite(UnityControl,sendbuf);  
        end
        if RandomTrial(AllTrial)==3  % ��̨������
            sendbuf(1,1) = hex2dec('02') ;
            sendbuf(1,2) = hex2dec('01') ;
            sendbuf(1,3) = hex2dec('00') ;
            sendbuf(1,4) = hex2dec('02') ;
            fwrite(UnityControl,sendbuf);  
        end
        if RandomTrial(AllTrial)==4  % ��̨������
            sendbuf(1,1) = hex2dec('02') ;
            sendbuf(1,2) = hex2dec('01') ;
            sendbuf(1,3) = hex2dec('00') ;
            sendbuf(1,4) = hex2dec('03') ;
            fwrite(UnityControl,sendbuf);  
        end
    end
    if Timer==5  %��ʼ���ǩ
        if RandomTrial(AllTrial)==1  % ��������
            Trigger = 1;
        end
        if RandomTrial(AllTrial)==2  % ��������
            Trigger = 2;
        end
        if RandomTrial(AllTrial)==3  % ��̨������
            Trigger = 3;
        end
        if RandomTrial(AllTrial)==4  % ��̨������
            Trigger = 4;
        end
    end
    if Timer==9  %��ʼ��Ϣ
        Trigger = 6;
        sendbuf(1,1) = hex2dec('04') ;
        sendbuf(1,2) = hex2dec('00') ;
        sendbuf(1,3) = hex2dec('00') ;
        sendbuf(1,4) = hex2dec('00') ;
        fwrite(UnityControl,sendbuf);  
    end

    tic
    fprintf('AllTrial = %d, Timer = %d.\n',AllTrial,Timer);
    [~, data] = ClientGetDataPacket(con,basicInfo,infoList,startStop,init); % Obtain EEG data, ��Ҫ��ClientGetDataPacket����Ҫ��Ҫ�Ƴ�����
% 	pause(1);
    fprintf('��ȡ 1 ���Ե����ݣ�');
    toc
    
    TriggerRepeat = repmat(Trigger,1,256);  
    data = [data;TriggerRepeat];
    TrialData = [TrialData,data];
    Timer = Timer + 1;
     
    if Timer == 11 % Start the next trial
        Timer = 0;
    end
    
end
%% �洢����
system('taskkill /F /IM ClimbStair3.exe');
system('taskkill /F /IM fes.exe');
close all;
TrialData = TrialData(2:end,:);  %ȥ�������һ��
ChanLabel = flip({infoList.chanLabel});
pnet('closeall');   % �����ӹر�

save(FunctionNowFilename('Offline_EEGdata_', '.mat' ),'TrialData','TrialIndex','ChanLabel');
