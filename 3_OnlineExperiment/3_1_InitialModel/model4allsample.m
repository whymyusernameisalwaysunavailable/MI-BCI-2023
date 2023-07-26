function [CSPs, svmModels] = model4allsample(FilterSample_Idle,FilterSample_Walk,FilterSample_Ascend,FilterSample_Descend,cmds,number_bandpass_filters,FilterNum)
    tic;
    assert(size(FilterSample_Idle,2) == size(FilterSample_Walk,2)...
            && size(FilterSample_Walk,2) == size(FilterSample_Ascend,2)...
            && size(FilterSample_Ascend,2) == size(FilterSample_Descend,2),...
            'FilterSample number error!!');
        
    load('fband_selection.mat');
    
    windows_per_action = size(FilterSample_Idle,2);

    FilterSample_Idle_train = FilterSample_Idle;
    FilterSample_Walk_train = FilterSample_Walk;
    FilterSample_Ascend_train = FilterSample_Ascend;
    FilterSample_Descend_train = FilterSample_Descend;
    samplenum_train = windows_per_action;
    
    FilterSample_train_Idle = repmat(FilterSample_Idle_train,1,3);
    FilterSample_train_notIdle = [FilterSample_Walk_train,FilterSample_Ascend_train,FilterSample_Descend_train];

                    
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
    toc;

    %% CSP Feature extraction for generating classifier
    fprintf('  生成所有样本特征及标签...');
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
    train_Fea_1 = [FeaSample_tr_Idle;FeaSample_tr_notIdle]; 
    train_Fea_Y_1 = [ones(samplenum_train*3,1);ones(samplenum_train*3,1)*2];

    % Walk vs. Ascend
    train_Fea_23 = [FeaSample_Walk_train_23;FeaSample_Ascend_train_23];
    train_Fea_Y_23 = [ones(samplenum_train,1);ones(samplenum_train,1)*2];

    % Walk vs. Descend
    train_Fea_24 = [FeaSample_Walk_train_24;FeaSample_Descend_train_24];
    train_Fea_Y_24 = [ones(samplenum_train,1);ones(samplenum_train,1)*2];

    % Ascend vs. Descend
    train_Fea_34 = [FeaSample_Ascend_train_34;FeaSample_Descend_train_34];
    train_Fea_Y_34 = [ones(samplenum_train,1);ones(samplenum_train,1)*2];
    toc;

    %% Classifier
    fprintf('  生成svm模型...');
    cmd_1 = cmds{1};
    cmd_2 = cmds{2};
    cmd_3 = cmds{3};
    cmd_4 = cmds{4};
    model_1 = svmtrain(train_Fea_Y_1,train_Fea_1(:,fband_mdl_1),cmd_1);
    model_23 = svmtrain(train_Fea_Y_23,train_Fea_23(:,fband_mdl_23),cmd_2);
    model_24 = svmtrain(train_Fea_Y_24,train_Fea_24(:,fband_mdl_24),cmd_3);
    model_34 = svmtrain(train_Fea_Y_34,train_Fea_34(:,fband_mdl_34),cmd_4);
    svmModels = struct('Mdl_1', model_1, 'Mdl_23', model_23, 'Mdl_24', model_24, 'Mdl_34', model_34,...
            'cmd_1', cmd_1,'cmd_2', cmd_2,'cmd_3', cmd_3,'cmd_4', cmd_4);
    toc;
end
    