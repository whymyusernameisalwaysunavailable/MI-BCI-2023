function [Feas, Y] = Class4_Online_Main_DataPreprocess_ov3ovo(online_new_rawSlideData,sample_frequency,channels,Wband)
    %% Parameters
    % Channel Selection
    number_of_channels = length(channels);
    
    % Sub-Fre Analysis
    number_bandpass_filters = size(Wband,1);
    FilterType = 'bandpass';
    FilterOrder = 4;
    
    %% Extracting label
    Tag = online_new_rawSlideData(end,:);
    assert(length(unique(Tag))==1,'Last 2 sec doesnt have unique label!');
    Y = unique(Tag);
    newSlideData = double(online_new_rawSlideData(1:end-1,Tag==unique(Tag)));
    
    %% Normalization
    signal = newSlideData;
    signal(channels,:) = signal(channels,:) - mean(signal(channels,:));
    signal(1:end,:) = signal(1:end,:)./var(signal(1:end,:),0,2);
    newSlideSample = signal(channels,:);
    
    %% SubFre filtering
    % shape: (number of bandpass filters, windows per action)
    newFilterSample = cell(number_bandpass_filters, 1);                       %2 bands * 1 cell, each cell is Channels * #of datapoint
    for i = 1:number_bandpass_filters
        newFilterSample{i,1} = Rsx_ButterFilter(FilterOrder,Wband(i,:),sample_frequency,FilterType,newSlideSample,number_of_channels);
    end
    
    %% CSP Feature Extraction
    load('ov3ovo_CspTranspose_forUse.mat');
    % shape: (windows per action, number of bandpass filters * FilterNum)
    Fea_1 = [];
    Fea_23 = [];
    Fea_24 = [];
    Fea_34 = [];
    for m =1:number_bandpass_filters
        Fea_1 = [Fea_1,Rsx_singlewindow_cspfeature(newFilterSample{m,1},CSPs{1}{m},FilterNum)];
        Fea_23 = [Fea_23,Rsx_singlewindow_cspfeature(newFilterSample{m,1},CSPs{2}{m},FilterNum)];
        Fea_24 = [Fea_24,Rsx_singlewindow_cspfeature(newFilterSample{m,1},CSPs{3}{m},FilterNum)];
        Fea_34 = [Fea_34,Rsx_singlewindow_cspfeature(newFilterSample{m,1},CSPs{4}{m},FilterNum)];
    end
    Feas = cell(1,4);
    Feas{1} = Fea_1;
    Feas{2} = Fea_23;
    Feas{3} = Fea_24;
    Feas{4} = Fea_34;
end


