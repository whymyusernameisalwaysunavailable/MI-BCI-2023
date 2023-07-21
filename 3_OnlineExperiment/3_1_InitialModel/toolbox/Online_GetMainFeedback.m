function Online_GetMainFeedback(MatlabComm_ModelUpdate,~,bytesToRead1,~,~)

global data_MainStat_collect;
bytesReady = MatlabComm_ModelUpdate.BytesAvailable;

if (bytesReady == 0)
    return
end
data_MainStat_collect = cast(fread(MatlabComm_ModelUpdate,bytesToRead1), 'uint8');

global dataReceive_FromMain;

dataReceive_FromMain = [dataReceive_FromMain, data_MainStat_collect];

end