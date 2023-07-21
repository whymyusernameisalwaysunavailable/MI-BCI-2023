function Online_GetModelUpdateFeedback(MatlabComm_Main,~,bytesToRead2,~,~)

global data_ModelUpdateStat_collect;
bytesReady = MatlabComm_Main.BytesAvailable;

if (bytesReady == 0)
    return
end
data_ModelUpdateStat_collect = cast(fread(MatlabComm_Main,bytesToRead2), 'uint8');

global dataReceive_FromModelUpdate;

dataReceive_FromModelUpdate = [dataReceive_FromModelUpdate, data_ModelUpdateStat_collect];

end