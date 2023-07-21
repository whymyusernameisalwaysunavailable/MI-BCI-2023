function Online_GetUnityFeedback(UnityControl,~,bytesToRead3,~,~)

global data_arrayEEG_collect;
bytesReady = UnityControl.BytesAvailable;

if (bytesReady == 0)
    return
end
data_arrayEEG_collect = cast(fread(UnityControl,bytesToRead3), 'uint8');

global dataReceive_FromUnity;

dataReceive_FromUnity = [dataReceive_FromUnity, data_arrayEEG_collect];

%fwrite(UnityControl,255);
end