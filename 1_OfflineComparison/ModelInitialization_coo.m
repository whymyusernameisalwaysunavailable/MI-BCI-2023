% Model initialization
%   2 step co-optimization of hyperparameters
%   Generate model based on data collected offline

addpath(genpath('toolbox'));
pnet('closeall');fclose all;clc;clear;close all;

%% Choose subjects for analysing
load('.\offlinedata-14-raw\subjectinfo.mat','Subject');

ACC_valtrain = [];
ACC_testtrain = [];
ACC_valval = [];
ACC_testtest = [];

for subjectnum = 1:length(Subject)

    offlinepath = ['.\offlinedata-14-raw\Subject_',num2str(Subject(subjectnum).number)];
    files = dir([offlinepath,'\FeaSets_*.mat']);
    valfoldnum = length(files);

    %% Parameters
    load([offlinepath,'\InitializationParameters.mat']);
    % data sample param
    data_points_per_TrainTrial = sample_frequency * seconds_per_TrainTrial;
    windows_per_trial = (data_points_per_TrainTrial - WindowLength) / SlideWindowLength + 1; % = 3
    windows_per_action = windows_per_trial * trials_per_action;                
    number_of_channels = length(channels);
    number_bandpass_filters = size(Wband,1); 

    % choose CSP param
    FilterNum = 4;% 不能超过通道数

    fea_num = number_bandpass_filters * FilterNum;
    
    %% Load & save pre-generated feature sets under current directory
    for valcount = 1:valfoldnum
        load([offlinepath,'\FeaSets_',num2str(valcount),'_for_classifiers_trte.mat'],...
                'VAL_train_Fea_1','VAL_train_Fea_Y_1',...
                'VAL_train_Fea_23','VAL_train_Fea_Y_23',...
                'VAL_train_Fea_24','VAL_train_Fea_Y_24',...
                'VAL_train_Fea_34','VAL_train_Fea_Y_34',...
                'VAL_val_Fea_all','VAL_val_Fea_Y');
        save(['FeaSets_',num2str(valcount),'_for_classifiers_trte.mat'],...
                'VAL_train_Fea_1','VAL_train_Fea_Y_1',...
                'VAL_train_Fea_23','VAL_train_Fea_Y_23',...
                'VAL_train_Fea_24','VAL_train_Fea_Y_24',...
                'VAL_train_Fea_34','VAL_train_Fea_Y_34',...
                'VAL_val_Fea_all','VAL_val_Fea_Y');
    end

    %% Co-optimization
    fprintf('  pso搜索最佳模型训练参数...');
    tic;
    num_stop_1=0;
    num_stop_2=0;
    num_stop_3=0;
    num_stop_4=0;
    num_particle_1=20;
    num_particle_2=20;
    maxOptLowerBound_1=75;
    maxOptLowerBound_2=75;
    maxOptLowerBound_3=75;
    maxOptLowerBound_4=75;
    num_stop_UpperBound=200;
    num_stop = 500000;
    
    CGKFbandScope=[1 3;-12 3;1 2^fea_num-0.01;0.3 0.3];
    ParticleSize = size(CGKFbandScope,1);
    [info_bestclassifier_1,OptSwarm_1,minmax_1,num_stop_1,k_1]=PsoProcessforCGKFband(num_particle_1,ParticleSize,CGKFbandScope,...
                    1,@InitCGSwarm,@BaseStepPso,@CGKFbandperformanceGray,0,0,num_stop,0,num_stop_1,maxOptLowerBound_1,num_stop_UpperBound);

    fband_1 = info_bestclassifier_1(3);
    fband_graycode_1 = dec2gray(floor(fband_1),fea_num);
    fband_mdl_1 = false(1,fea_num);
    fband_mdl_1(1,:) = flipud(logical(str2num(fband_graycode_1(:))));
    save('fband_selection.mat','fband_mdl_1');

    CGKFbandScope=[1 3;-12 3;0.1 0.1;0.3 0.3];
    ParticleSize = size(CGKFbandScope,1);
    [info_bestclassifier_1,OptSwarm_1,minmax_1,num_stop_1,k_1]=PsoProcessforCGKFband(num_particle_1,ParticleSize,CGKFbandScope,...
                    1,@InitCGSwarm,@BaseStepPso,@CGKFbandperformanceGray,0,0,num_stop,0,num_stop_1,maxOptLowerBound_1,num_stop_UpperBound);


    t1 = toc;

    CGKFbandScope=[1 3;-12 3;1 2^fea_num-0.01;0.3 0.3];
    ParticleSize = size(CGKFbandScope,1);
    [info_bestclassifier_2,OptSwarm_2,minmax_2,num_stop_2,k_2]=PsoProcessforCGKFband(num_particle_2,ParticleSize,CGKFbandScope,...
                    2,@InitCGSwarm,@BaseStepPso,@CGKFbandperformanceGray,0,0,num_stop,0,num_stop_2,maxOptLowerBound_2,num_stop_UpperBound);

    fband_2 = info_bestclassifier_2(3);
    fband_graycode_2 = dec2gray(floor(fband_2),fea_num);
    fband_mdl_23 = false(1,fea_num);
    fband_mdl_23(1,:) = flipud(logical(str2num(fband_graycode_2(:))));
    save('fband_selection.mat','fband_mdl_23','-append');

    CGKFbandScope=[1 3;-12 3;0.1 0.1;0.3 0.3];
    ParticleSize = size(CGKFbandScope,1);
    [info_bestclassifier_2,OptSwarm_2,minmax_2,num_stop_2,k_2]=PsoProcessforCGKFband(num_particle_2,ParticleSize,CGKFbandScope,...
                    2,@InitCGSwarm,@BaseStepPso,@CGKFbandperformanceGray,0,0,num_stop,0,num_stop_2,maxOptLowerBound_2,num_stop_UpperBound);


    t2 = toc-t1;
    
    CGKFbandScope=[1 3;-12 3;1 2^fea_num-0.01;0.3 0.3];
    ParticleSize = size(CGKFbandScope,1);
    [info_bestclassifier_3,OptSwarm_3,minmax_3,num_stop_3,k_3]=PsoProcessforCGKFband(num_particle_2,ParticleSize,CGKFbandScope,...
                    3,@InitCGSwarm,@BaseStepPso,@CGKFbandperformanceGray,0,0,num_stop,0,num_stop_3,maxOptLowerBound_3,num_stop_UpperBound);

    fband_3 = info_bestclassifier_3(3);
    fband_graycode_3 = dec2gray(floor(fband_3),fea_num);
    fband_mdl_24 = false(1,fea_num);
    fband_mdl_24(1,:) = flipud(logical(str2num(fband_graycode_3(:))));
    save('fband_selection.mat','fband_mdl_24','-append');

    CGKFbandScope=[1 3;-12 3;0.1 0.1;0.3 0.3];
    ParticleSize = size(CGKFbandScope,1);
    [info_bestclassifier_3,OptSwarm_3,minmax_3,num_stop_3,k_3]=PsoProcessforCGKFband(num_particle_2,ParticleSize,CGKFbandScope,...
                    3,@InitCGSwarm,@BaseStepPso,@CGKFbandperformanceGray,0,0,num_stop,0,num_stop_3,maxOptLowerBound_3,num_stop_UpperBound);


    t3 = toc-t2-t1;
    
    CGKFbandScope=[1 3;-12 3;1 2^fea_num-0.01;0.3 0.3];
    ParticleSize = size(CGKFbandScope,1);
    [info_bestclassifier_4,OptSwarm_4,minmax_4,num_stop_4,k_4]=PsoProcessforCGKFband(num_particle_2,ParticleSize,CGKFbandScope,...
                    4,@InitCGSwarm,@BaseStepPso,@CGKFbandperformanceGray,0,0,num_stop,0,num_stop_4,maxOptLowerBound_4,num_stop_UpperBound);

    fband_4 = info_bestclassifier_4(3);
    fband_graycode_4 = dec2gray(floor(fband_4),fea_num);
    fband_mdl_34 = false(1,fea_num);
    fband_mdl_34(1,:) = flipud(logical(str2num(fband_graycode_4(:))));
    save('fband_selection.mat','fband_mdl_34','-append');

    CGKFbandScope=[1 3;-12 3;0.1 0.1;0.3 0.3];
    ParticleSize = size(CGKFbandScope,1);
    [info_bestclassifier_4,OptSwarm_4,minmax_4,num_stop_4,k_4]=PsoProcessforCGKFband(num_particle_2,ParticleSize,CGKFbandScope,...
                    4,@InitCGSwarm,@BaseStepPso,@CGKFbandperformanceGray,0,0,num_stop,0,num_stop_4,maxOptLowerBound_4,num_stop_UpperBound);


    t4 = toc-t3-t2-t1;
    psotime = toc;

    cmd_1 = ['-c ',num2str(2^info_bestclassifier_1(1)),' -g ',num2str(2^info_bestclassifier_1(2)),' -b 1'];
    cmd_2 = ['-c ',num2str(2^info_bestclassifier_2(1)),' -g ',num2str(2^info_bestclassifier_2(2)),' -b 1'];
    cmd_3 = ['-c ',num2str(2^info_bestclassifier_3(1)),' -g ',num2str(2^info_bestclassifier_3(2)),' -b 1'];
    cmd_4 = ['-c ',num2str(2^info_bestclassifier_4(1)),' -g ',num2str(2^info_bestclassifier_4(2)),' -b 1'];

    save([offlinepath,'\Optcgf_',num2str(Subject(subjectnum).number), '.mat'],...
            'fband_1','fband_2','fband_3','fband_4',...
            'info_bestclassifier_1','info_bestclassifier_2',...
            'info_bestclassifier_3','info_bestclassifier_4',...
            'OptSwarm_1','OptSwarm_2','OptSwarm_3','OptSwarm_4',...
            't1','t2','t3','t4','psotime',...
            'fband_mdl_1','fband_mdl_23','fband_mdl_24','fband_mdl_34',...
            'cmd_1','cmd_2','cmd_3','cmd_4');
        
    %% Validation & Test
    VAL_train_acc_mdls = zeros(4,valfoldnum+1);
    % 4模型各自的二分类准确率，为五次训练-测试平均值
    VAL_val_acc_mdls = zeros(4,valfoldnum+1);

    % 总预测结果的五次正确率，及其平均
    VAL_val_acc_all = zeros(1,valfoldnum+1);
    val_acc_all_tmp = 0;

    % 四行分别对应四类各自的最终预测结果正确率（五次），最后一列为五次平均值
    VAL_val_recall_4classes = zeros(4,valfoldnum+1);
    VAL_val_precision_4classes = zeros(4,valfoldnum+1);
    VAL_val_precision_all = zeros(1,valfoldnum+1);

    % F1-score = 2/(1/recall + 1/precision)
    VAL_val_F1_score_4classes = zeros(4,valfoldnum+1);
    VAL_val_F1_score_all = zeros(1,valfoldnum+1);

    for valcount = 1:valfoldnum
        load(['FeaSets_',num2str(valcount),'_for_classifiers_trte.mat']);

        model_1 = svmtrain(VAL_train_Fea_Y_1,VAL_train_Fea_1(:,fband_mdl_1),cmd_1);
        model_23 = svmtrain(VAL_train_Fea_Y_23,VAL_train_Fea_23(:,fband_mdl_23),cmd_2);
        model_24 = svmtrain(VAL_train_Fea_Y_24,VAL_train_Fea_24(:,fband_mdl_24),cmd_3);
        model_34 = svmtrain(VAL_train_Fea_Y_34,VAL_train_Fea_34(:,fband_mdl_34),cmd_4);
        svmModels = struct('Mdl_1', model_1, 'Mdl_23', model_23, 'Mdl_24', model_24, 'Mdl_34', model_34,...
                'cmd_1', cmd_1,'cmd_2', cmd_2,'cmd_3', cmd_3,'cmd_4', cmd_4);

        [predict_label_te_1,accuracy_te_1,prob_estimates_te] = svmpredict(VAL_train_Fea_Y_1,VAL_train_Fea_1(:,fband_mdl_1),model_1,' -b 1');
        [predict_label_te_2,accuracy_te_2,prob_estimates_te] = svmpredict(VAL_train_Fea_Y_23,VAL_train_Fea_23(:,fband_mdl_23),model_23,' -b 1');
        [predict_label_te_3,accuracy_te_3,prob_estimates_te] = svmpredict(VAL_train_Fea_Y_24,VAL_train_Fea_24(:,fband_mdl_24),model_24,' -b 1');
        [predict_label_te_4,accuracy_te_4,prob_estimates_te] = svmpredict(VAL_train_Fea_Y_34,VAL_train_Fea_34(:,fband_mdl_34),model_34,' -b 1');

        VAL_train_acc_mdls(1,valcount) = accuracy_te_1(1);
        VAL_train_acc_mdls(2,valcount) = accuracy_te_2(1);
        VAL_train_acc_mdls(3,valcount) = accuracy_te_3(1);
        VAL_train_acc_mdls(4,valcount) = accuracy_te_4(1);  

        predict_label_val = svmPredict_4Class_ov3ovo_selectfband(VAL_val_Fea_all,VAL_val_Fea_Y,svmModels);

        VAL_val_acc_all(valcount) = sum(predict_label_val == VAL_val_Fea_Y)/size(VAL_val_Fea_Y,1);

        isinval_init = (VAL_val_Fea_all{1}(:,1) ~= 0);
        isinval_Idle = isinval_init(1:3);
        isinval_Walk = isinval_init(4:6);
        isinval_Ascend = isinval_init(7:9);
        isinval_Descend = isinval_init(10:12);

        val_Fea_isIdle = repmat(VAL_val_Fea_all{1}(1:3,fband_mdl_1),3,1);
        val_Fea_notIdle = VAL_val_Fea_all{1}(4:12,fband_mdl_1);
        index_val_1 = repmat(isinval_Idle,3,1) & [isinval_Walk;...
                                              isinval_Ascend;...
                                              isinval_Descend];
        VAL_val_Fea_1 = [val_Fea_isIdle(index_val_1,:);val_Fea_notIdle(index_val_1,:)];
        VAL_val_Fea_Y_1 = [ones(sum(index_val_1),1);ones(sum(index_val_1),1)*2];

        val_Fea_23_tmp = VAL_val_Fea_all{2}(4:9,fband_mdl_23);
        index_val_2 = isinval_Walk & isinval_Ascend;
        VAL_val_Fea_2 = val_Fea_23_tmp([index_val_2;index_val_2],:);
        VAL_val_Fea_Y_2 = [ones(sum(index_val_2),1);ones(sum(index_val_2),1)*2];

        val_Fea_24_tmp = VAL_val_Fea_all{3}([4:6,10:12],fband_mdl_24);
        index_val_3 = isinval_Walk & isinval_Descend;
        VAL_val_Fea_3 = val_Fea_24_tmp([index_val_3;index_val_3],:);
        VAL_val_Fea_Y_3 = [ones(sum(index_val_3),1);ones(sum(index_val_3),1)*2];

        val_Fea_34_tmp = VAL_val_Fea_all{4}(7:12,fband_mdl_34);
        index_val_4 = isinval_Ascend & isinval_Descend;
        VAL_val_Fea_4 = val_Fea_34_tmp([index_val_4;index_val_4],:);
        VAL_val_Fea_Y_4 = [ones(sum(index_val_4),1);ones(sum(index_val_4),1)*2];

        [predict_label_te_1,accuracy_te_1,prob_estimates_val] = svmpredict(VAL_val_Fea_Y_1,VAL_val_Fea_1,model_1,' -b 1');
        [predict_label_te_2,accuracy_te_2,prob_estimates_val] = svmpredict(VAL_val_Fea_Y_2,VAL_val_Fea_2,model_23,' -b 1');
        [predict_label_te_3,accuracy_te_3,prob_estimates_val] = svmpredict(VAL_val_Fea_Y_3,VAL_val_Fea_3,model_24,' -b 1');
        [predict_label_te_4,accuracy_te_4,prob_estimates_val] = svmpredict(VAL_val_Fea_Y_4,VAL_val_Fea_4,model_34,' -b 1');

        VAL_val_acc_mdls(1,valcount) = accuracy_te_1(1);
        VAL_val_acc_mdls(2,valcount) = accuracy_te_2(1);
        VAL_val_acc_mdls(3,valcount) = accuracy_te_3(1);
        VAL_val_acc_mdls(4,valcount) = accuracy_te_4(1);  

        predict_label_tmp = svmPredict_4Class_ov3ovo_selectfband(VAL_val_Fea_all,VAL_val_Fea_Y,svmModels);
        predict_label_val = predict_label_tmp(isinval_init);
        VAL_val_Fea_Y_val = VAL_val_Fea_Y(isinval_init);

        VAL_val_acc_all(valcount) = sum(predict_label_val == VAL_val_Fea_Y_val)/size(VAL_val_Fea_Y_val,1);
        VAL_val_recall_4classes(1,valcount) = sum(predict_label_val(VAL_val_Fea_Y_val==1,1) == 1)/sum(VAL_val_Fea_Y_val == 1);
        VAL_val_recall_4classes(2,valcount) = sum(predict_label_val(VAL_val_Fea_Y_val==2,1) == 2)/sum(VAL_val_Fea_Y_val == 2);
        VAL_val_recall_4classes(3,valcount) = sum(predict_label_val(VAL_val_Fea_Y_val==3,1) == 3)/sum(VAL_val_Fea_Y_val == 3);
        VAL_val_recall_4classes(4,valcount) = sum(predict_label_val(VAL_val_Fea_Y_val==4,1) == 4)/sum(VAL_val_Fea_Y_val == 4);   

        VAL_val_precision_4classes(1,valcount) = sum(VAL_val_Fea_Y_val(predict_label_val==1,1) == 1)/sum(predict_label_val == 1);
        VAL_val_precision_4classes(2,valcount) = sum(VAL_val_Fea_Y_val(predict_label_val==2,1) == 2)/sum(predict_label_val == 2);
        VAL_val_precision_4classes(3,valcount) = sum(VAL_val_Fea_Y_val(predict_label_val==3,1) == 3)/sum(predict_label_val == 3);
        VAL_val_precision_4classes(4,valcount) = sum(VAL_val_Fea_Y_val(predict_label_val==4,1) == 4)/sum(predict_label_val == 4);   
    
    end

    VAL_train_acc_mdls(:,valfoldnum+1) = mean(VAL_train_acc_mdls(:,1:valfoldnum),2);

    VAL_val_acc_all(valfoldnum+1) = mean(VAL_val_acc_all(1:valfoldnum));
    VAL_val_acc_mdls(:,valfoldnum+1) = nanmean(VAL_val_acc_mdls(:,1:valfoldnum),2);
    VAL_val_acc_all(valfoldnum+1) = mean(VAL_val_acc_all(1:valfoldnum));
    VAL_val_recall_4classes(:,valfoldnum+1) = nanmean(VAL_val_recall_4classes(:,1:valfoldnum),2);
    VAL_val_precision_4classes(:,valfoldnum+1) = nanmean(VAL_val_precision_4classes(:,1:valfoldnum),2);
    VAL_val_precision_all(valfoldnum+1) = nanmean(VAL_val_precision_4classes(:,valfoldnum+1));
    VAL_val_F1_score_4classes(:,valfoldnum+1) = 2./(1./VAL_val_recall_4classes(:,valfoldnum+1) + 1./VAL_val_precision_4classes(:,valfoldnum+1));
    VAL_val_F1_score_all(valfoldnum+1) = 2./(1./VAL_val_acc_all(valfoldnum+1) + 1./VAL_val_precision_all(valfoldnum+1));

    VAL_val_score = VAL_val_acc_mdls(:,valfoldnum+1)- 0.3 * abs(VAL_val_acc_mdls(:,valfoldnum+1)-VAL_train_acc_mdls(:,valfoldnum+1));
    VAL_val_all = [VAL_val_acc_mdls(:,valfoldnum+1);...
                   VAL_val_score;...
                   VAL_val_acc_all(valfoldnum+1);...
                   VAL_val_recall_4classes(:,valfoldnum+1);...
                   VAL_val_precision_4classes(:,valfoldnum+1);...
                   VAL_val_precision_all(valfoldnum+1);...
                   VAL_val_F1_score_4classes(:,valfoldnum+1);...
                   VAL_val_F1_score_all(valfoldnum+1)];
    
    TEST_train_acc_mdls = zeros(4,valfoldnum+1);
    TEST_test_acc_mdls = zeros(4,valfoldnum+1);
    TEST_test_acc_all = zeros(1,valfoldnum+1);
    TEST_test_recall_4classes = zeros(4,valfoldnum+1);
    TEST_test_precision_4classes = zeros(4,valfoldnum+1);
    TEST_test_precision_all = zeros(1,valfoldnum+1);
    TEST_test_F1_score_4classes = zeros(4,valfoldnum+1);
    TEST_test_F1_score_all = zeros(1,valfoldnum+1);
   
    for valcount = 1:valfoldnum
        load([offlinepath,'\FeaSets_',num2str(valcount),'_for_classifiers_trte.mat']);

        model_1_mdl = svmtrain(TEST_train_Fea_Y_1,TEST_train_Fea_1(:,fband_mdl_1),cmd_1);
        model_23_mdl = svmtrain(TEST_train_Fea_Y_23,TEST_train_Fea_23(:,fband_mdl_23),cmd_2);
        model_24_mdl = svmtrain(TEST_train_Fea_Y_24,TEST_train_Fea_24(:,fband_mdl_24),cmd_3);
        model_34_mdl = svmtrain(TEST_train_Fea_Y_34,TEST_train_Fea_34(:,fband_mdl_34),cmd_4);
        svmModels_mdl = struct('Mdl_1', model_1_mdl, 'Mdl_23', model_23_mdl, 'Mdl_24', model_24_mdl, 'Mdl_34', model_34_mdl,...
                'cmd_1', cmd_1,'cmd_2', cmd_2,'cmd_3', cmd_3,'cmd_4', cmd_4);
            
        [predict_label_te_1,accuracy_te_1,prob_estimates_te] = svmpredict(TEST_train_Fea_Y_1,TEST_train_Fea_1(:,fband_mdl_1),model_1_mdl,' -b 1');
        [predict_label_te_2,accuracy_te_2,prob_estimates_te] = svmpredict(TEST_train_Fea_Y_23,TEST_train_Fea_23(:,fband_mdl_23),model_23_mdl,' -b 1');
        [predict_label_te_3,accuracy_te_3,prob_estimates_te] = svmpredict(TEST_train_Fea_Y_24,TEST_train_Fea_24(:,fband_mdl_24),model_24_mdl,' -b 1');
        [predict_label_te_4,accuracy_te_4,prob_estimates_te] = svmpredict(TEST_train_Fea_Y_34,TEST_train_Fea_34(:,fband_mdl_34),model_34_mdl,' -b 1');

        TEST_train_acc_mdls(1,valcount) = accuracy_te_1(1);
        TEST_train_acc_mdls(2,valcount) = accuracy_te_2(1);
        TEST_train_acc_mdls(3,valcount) = accuracy_te_3(1);
        TEST_train_acc_mdls(4,valcount) = accuracy_te_4(1);  

        isintest_init = (TEST_test_Fea_all{1}(:,1) ~= 0);
        isintest_Idle = isintest_init(1:3);
        isintest_Walk = isintest_init(4:6);
        isintest_Ascend = isintest_init(7:9);
        isintest_Descend = isintest_init(10:12);

        test_Fea_isIdle = repmat(TEST_test_Fea_all{1}(1:3,fband_mdl_1),3,1);
        test_Fea_notIdle = TEST_test_Fea_all{1}(4:12,fband_mdl_1);
        index_test_1 = repmat(isintest_Idle,3,1) & [isintest_Walk;...
                                              isintest_Ascend;...
                                              isintest_Descend];
        TEST_test_Fea_1 = [test_Fea_isIdle(index_test_1,:);test_Fea_notIdle(index_test_1,:)];
        TEST_test_Fea_Y_1 = [ones(sum(index_test_1),1);ones(sum(index_test_1),1)*2];

        test_Fea_23_tmp = TEST_test_Fea_all{2}(4:9,fband_mdl_23);
        index_test_2 = isintest_Walk & isintest_Ascend;
        TEST_test_Fea_2 = test_Fea_23_tmp([index_test_2;index_test_2],:);
        TEST_test_Fea_Y_2 = [ones(sum(index_test_2),1);ones(sum(index_test_2),1)*2];

        test_Fea_24_tmp = TEST_test_Fea_all{3}([4:6,10:12],fband_mdl_24);
        index_test_3 = isintest_Walk & isintest_Descend;
        TEST_test_Fea_3 = test_Fea_24_tmp([index_test_3;index_test_3],:);
        TEST_test_Fea_Y_3 = [ones(sum(index_test_3),1);ones(sum(index_test_3),1)*2];

        test_Fea_34_tmp = TEST_test_Fea_all{4}(7:12,fband_mdl_34);
        index_test_4 = isintest_Ascend & isintest_Descend;
        TEST_test_Fea_4 = test_Fea_34_tmp([index_test_4;index_test_4],:);
        TEST_test_Fea_Y_4 = [ones(sum(index_test_4),1);ones(sum(index_test_4),1)*2];

        [predict_label_te_1,accuracy_te_1,prob_estimates_te] = svmpredict(TEST_test_Fea_Y_1,TEST_test_Fea_1,model_1_mdl,' -b 1');
        [predict_label_te_2,accuracy_te_2,prob_estimates_te] = svmpredict(TEST_test_Fea_Y_2,TEST_test_Fea_2,model_23_mdl,' -b 1');
        [predict_label_te_3,accuracy_te_3,prob_estimates_te] = svmpredict(TEST_test_Fea_Y_3,TEST_test_Fea_3,model_24_mdl,' -b 1');
        [predict_label_te_4,accuracy_te_4,prob_estimates_te] = svmpredict(TEST_test_Fea_Y_4,TEST_test_Fea_4,model_34_mdl,' -b 1');

        TEST_test_acc_mdls(1,valcount) = accuracy_te_1(1);
        TEST_test_acc_mdls(2,valcount) = accuracy_te_2(1);
        TEST_test_acc_mdls(3,valcount) = accuracy_te_3(1);
        TEST_test_acc_mdls(4,valcount) = accuracy_te_4(1);  

        predict_label_tmp = svmPredict_4Class_ov3ovo_selectfband(TEST_test_Fea_all,TEST_test_Fea_Y,svmModels_mdl);
        predict_label_test = predict_label_tmp(isintest_init);
        TEST_test_Fea_Y_test = TEST_test_Fea_Y(isintest_init);

        TEST_test_acc_all(valcount) = sum(predict_label_test == TEST_test_Fea_Y_test)/size(TEST_test_Fea_Y_test,1);
        TEST_test_recall_4classes(1,valcount) = sum(predict_label_test(TEST_test_Fea_Y_test==1,1) == 1)/sum(TEST_test_Fea_Y_test == 1);
        TEST_test_recall_4classes(2,valcount) = sum(predict_label_test(TEST_test_Fea_Y_test==2,1) == 2)/sum(TEST_test_Fea_Y_test == 2);
        TEST_test_recall_4classes(3,valcount) = sum(predict_label_test(TEST_test_Fea_Y_test==3,1) == 3)/sum(TEST_test_Fea_Y_test == 3);
        TEST_test_recall_4classes(4,valcount) = sum(predict_label_test(TEST_test_Fea_Y_test==4,1) == 4)/sum(TEST_test_Fea_Y_test == 4);   

        TEST_test_precision_4classes(1,valcount) = sum(TEST_test_Fea_Y_test(predict_label_test==1,1) == 1)/sum(predict_label_test == 1);
        TEST_test_precision_4classes(2,valcount) = sum(TEST_test_Fea_Y_test(predict_label_test==2,1) == 2)/sum(predict_label_test == 2);
        TEST_test_precision_4classes(3,valcount) = sum(TEST_test_Fea_Y_test(predict_label_test==3,1) == 3)/sum(predict_label_test == 3);
        TEST_test_precision_4classes(4,valcount) = sum(TEST_test_Fea_Y_test(predict_label_test==4,1) == 4)/sum(predict_label_test == 4);   

    end

    TEST_test_acc_mdls(:,valfoldnum+1) = nanmean(TEST_test_acc_mdls(:,1:valfoldnum),2);
    TEST_test_acc_all(valfoldnum+1) = mean(TEST_test_acc_all(1:valfoldnum));
    TEST_test_recall_4classes(:,valfoldnum+1) = nanmean(TEST_test_recall_4classes(:,1:valfoldnum),2);
    TEST_test_precision_4classes(:,valfoldnum+1) = nanmean(TEST_test_precision_4classes(:,1:valfoldnum),2);
    TEST_test_precision_all(valfoldnum+1) = nanmean(TEST_test_precision_4classes(:,valfoldnum+1));
    TEST_test_F1_score_4classes(:,valfoldnum+1) = 2./(1./TEST_test_recall_4classes(:,valfoldnum+1) + 1./TEST_test_precision_4classes(:,valfoldnum+1));
    TEST_test_F1_score_all(valfoldnum+1) = 2./(1./TEST_test_acc_all(valfoldnum+1) + 1./TEST_test_precision_all(valfoldnum+1));

    TEST_test_acc_mdls_sort = sort(TEST_test_acc_mdls(:,1:valfoldnum),2,'descend');
    TEST_test_acc_mdls_max3 = mean(TEST_test_acc_mdls_sort(:,1:3),2);
    TEST_test_recall_4classes_sort = sort(TEST_test_recall_4classes(:,1:valfoldnum),2,'descend');
    TEST_test_recall_4classes_max3 = mean(TEST_test_recall_4classes_sort(:,1:3),2);
    
    TEST_test_acc_all_sort = sort(TEST_test_acc_all(:,1:valfoldnum),2,'descend');
    TEST_test_acc_all_max3 = mean(TEST_test_acc_all_sort(:,1:3),2);

    TEST_test_all = [TEST_test_acc_mdls(:,valfoldnum+1);...
                     TEST_test_acc_all(valfoldnum+1);...
                     TEST_test_acc_mdls_max3;...
                     TEST_test_acc_all_max3;...
                     TEST_test_recall_4classes(:,valfoldnum+1);...
                     TEST_test_precision_4classes(:,valfoldnum+1);...
                     TEST_test_precision_all(valfoldnum+1);...
                     TEST_test_F1_score_4classes(:,valfoldnum+1);...
                     TEST_test_F1_score_all(valfoldnum+1)];
                 
    fmt = repmat('%2.4f ',1,valfoldnum);
    fprintf(['      test_acc_all      = ',fmt,'  average %.4f\n'],TEST_test_acc_all);
    fprintf('      test_acc_4classes = %2.4f %2.4f %2.4f %2.4f\n',TEST_test_recall_4classes(:,valfoldnum+1));
    fprintf('      test_acc_4mdls    = %2.4f %2.4f %2.4f %2.4f\n',TEST_test_acc_mdls(:,valfoldnum+1));

    ACC_valtrain = [ACC_valtrain;VAL_train_acc_mdls(:,valfoldnum+1)'];
    ACC_testtrain = [ACC_testtrain;TEST_train_acc_mdls(:,valfoldnum+1)'];

    ACC_valval = [ACC_valval;VAL_val_all'];
    ACC_testtest = [ACC_testtest;TEST_test_all'];
        
    save([offlinepath,'\result_',num2str(Subject(subjectnum).number), '_coo.mat'],...
        'VAL_train_acc_mdls',...
        'VAL_val_acc_mdls','VAL_val_acc_all','VAL_val_recall_4classes',...
        'VAL_val_precision_4classes','VAL_val_F1_score_4classes',...
        'VAL_val_all',...
        'TEST_train_acc_mdls',...
        'TEST_test_acc_mdls','TEST_test_acc_all','TEST_test_recall_4classes',...
        'TEST_test_precision_4classes','TEST_test_F1_score_4classes',...
        'TEST_test_all');

end

if size(ACC_valtrain,1)>1
    ACC_valtrain = [ACC_valtrain;nanmean(ACC_valtrain,1);nanstd(ACC_valtrain)];
    ACC_testtrain = [ACC_testtrain;nanmean(ACC_testtrain,1);nanstd(ACC_testtrain)];
    ACC_valval = [ACC_valval;nanmean(ACC_valval,1);nanstd(ACC_valval)];
    ACC_testtest = [ACC_testtest;nanmean(ACC_testtest,1);nanstd(ACC_testtest)];
end
save('ACC_all_coo.mat','ACC_valtrain','ACC_testtrain','ACC_valval','ACC_testtest','Subject');
