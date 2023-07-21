% Send all 60 trials' scenes to Unity
%% Log
% Updated Time:     2021/11/22
% Author:           Chutian Zhang(张楚天)
% Version:          V 1.0.0
% File:             Online_SendScenes.mat
% Describe:         Write during the PhD work at CASIA, Github link: https://github.com/JodyZZZ/MI-BCI
%% 
function [FinishSendScenesFlag]=Online_SendScenes(index,AllRandomTrial,TotalTrialNum)
    global UnityControl
%     global dataReceive_FromUnity;
    pause(1)
    while (index <= TotalTrialNum)
        if AllRandomTrial(index)==2  % 步行任务，停止，“ ”
            sendbuf(1,1) = hex2dec('02') ;
            sendbuf(1,2) = hex2dec('00') ;
            sendbuf(1,3) = hex2dec('00') ;
            sendbuf(1,4) = hex2dec('01') ;
            sendbuf(1,5) = hex2dec('00') ;
            fwrite(UnityControl,sendbuf); 
        end
        if AllRandomTrial(index)==3  % 上台阶任务，停止，“ ”
            sendbuf(1,1) = hex2dec('02') ;
            sendbuf(1,2) = hex2dec('00') ;
            sendbuf(1,3) = hex2dec('00') ;
            sendbuf(1,4) = hex2dec('02') ;
            sendbuf(1,5) = hex2dec('00') ;
            fwrite(UnityControl,sendbuf); 
        end
        if AllRandomTrial(index)==4  % 下台阶任务，停止，“ ”
            sendbuf(1,1) = hex2dec('02') ;
            sendbuf(1,2) = hex2dec('00') ;
            sendbuf(1,3) = hex2dec('00') ;
            sendbuf(1,4) = hex2dec('03') ;
            sendbuf(1,5) = hex2dec('00') ;
            fwrite(UnityControl,sendbuf); 
        end
        
        index = index + 1;
        pause(0.8)
    end
    if index == TotalTrialNum+1
        FinishSendScenesFlag = 1;
        
    else
        FinishSendScenesFlag = 0;
    end

    
    