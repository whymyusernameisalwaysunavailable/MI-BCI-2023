% Based on pre-extracted feature sets of offline eeg data
%
% 5-fold testing to evaluate the stepwiselinear model performance
%
% (10 vs. 1) * 5 times to test the performance of swlr model
% 10 vs. 1: Training: Folds No.6-15; Testing: Fold No.16;
%           Training: Folds No.7-16; Testing: Fold No.17;
%           Training: Folds No.8-17; Testing: Fold No.18;
%           Training: Folds No.9-18; Testing: Fold No.19;
%           Training: Folds No.10-19; Testing: Fold No.20;

addpath(genpath('toolbox'));
pnet('closeall');fclose all;clc;clear;close all;

%% Choosing subjects for analysing
load('.\offlinedata-14-raw\subjectinfo.mat','Subject');

ACC_train = [];
ACC_test = [];
for subjectnum = 1:length(Subject)
    offlinepath = ['.\offlinedata-14-raw\Subject_',num2str(Subject(subjectnum).number)];
    valfiles = dir([offlinepath,'\FeaSets_*.mat']);
    valfoldnum = length(valfiles);

    %% Parameters
    load([offlinepath,'\InitializationParameters.mat']);
    % data sample param
    data_points_per_TrainTrial = sample_frequency * seconds_per_TrainTrial;
    number_of_channels = length(channels);
    number_bandpass_filters = size(Wband,1); 

    % choose CSP param
    FilterNum = 4;% 不能超过通道数

%     %% Load & save pre-generated feature sets under current directory
%     for valcount = 1:valfoldnum
%         load([offlinepath,'\FeaSets_',num2str(valcount),'_for_classifiers_trte.mat'],...
%                 'VAL_train_Fea_1','VAL_train_Fea_Y_1',...
%                 'VAL_train_Fea_23','VAL_train_Fea_Y_23',...
%                 'VAL_train_Fea_24','VAL_train_Fea_Y_24',...
%                 'VAL_train_Fea_34','VAL_train_Fea_Y_34',...
%                 'VAL_val_Fea_all','VAL_val_Fea_Y');
%         save(['FeaSets_',num2str(valcount),'_for_classifiers_trte.mat'],...
%                 'VAL_train_Fea_1','VAL_train_Fea_Y_1',...
%                 'VAL_train_Fea_23','VAL_train_Fea_Y_23',...
%                 'VAL_train_Fea_24','VAL_train_Fea_Y_24',...
%                 'VAL_train_Fea_34','VAL_train_Fea_Y_34',...
%                 'VAL_val_Fea_all','VAL_val_Fea_Y');
%     end

    %% Classifier
    fprintf('  逐步线性回归...');
    tic;
        
    % 4模型各自的二分类准确率，为五次训练-测试平均值
    test_acc_mdls = zeros(4,valfoldnum+1);
    train_acc_mdls = zeros(4,valfoldnum+1);

    % 总预测结果的五次正确率，及五次平均
    test_acc_all = zeros(1,valfoldnum+1);
    
    % 四行分别对应四类各自的最终预测结果正确率（五次），最后一列为五次平均值
    test_recall_4classes = zeros(4,valfoldnum+1);
    test_precision_4classes = zeros(4,valfoldnum+1);
    test_precision_all = zeros(1,valfoldnum+1);

    % F1-score = 2/(1/recall + 1/precision)
    test_F1_score_4classes = zeros(4,valfoldnum+1);   
    test_F1_score_all = zeros(1,valfoldnum+1);
    
    for valcount = 1:valfoldnum
        load([offlinepath,'\FeaSets_',num2str(valcount),'_for_classifiers_trte.mat']);
        
        model_1 = stepwiselm(TEST_train_Fea_1, TEST_train_Fea_Y_1,'constant','upper','quadratic', 'Criterion', 'sse', 'PEnter', 0.05, 'PRemove', 0.1);
        model_23 = stepwiselm(TEST_train_Fea_23, TEST_train_Fea_Y_23,'constant','upper','quadratic', 'Criterion', 'sse', 'PEnter', 0.05, 'PRemove', 0.1);
        model_24 = stepwiselm(TEST_train_Fea_24, TEST_train_Fea_Y_24,'constant','upper','quadratic', 'Criterion', 'sse', 'PEnter', 0.05, 'PRemove', 0.1);
        model_34 = stepwiselm(TEST_train_Fea_34, TEST_train_Fea_Y_34,'constant','upper','quadratic', 'Criterion', 'sse', 'PEnter', 0.05, 'PRemove', 0.1);
        boundary_1 = 1.5;
        boundary_2 = 1.5;
        boundary_3 = 1.5;
        boundary_4 = 1.5;
        swlrModels = struct('Mdl_1', model_1, 'Mdl_23', model_23, 'Mdl_24', model_24, 'Mdl_34', model_34,...
            'boundary_1',boundary_1,'boundary_2',boundary_2,'boundary_3',boundary_3,'boundary_4',boundary_4);
        
        predictions_tr_1 = predict(model_1, TEST_train_Fea_1);
        predictions_tr_2 = predict(model_23, TEST_train_Fea_23);
        predictions_tr_3 = predict(model_24, TEST_train_Fea_24);
        predictions_tr_4 = predict(model_34, TEST_train_Fea_34);
                                 
        train_acc_mdls(1,valcount) = mean([sum(predictions_tr_1(TEST_train_Fea_Y_1==1) < boundary_1)/sum(TEST_train_Fea_Y_1 == 1),...
                                     sum(predictions_tr_1(TEST_train_Fea_Y_1==2) >= boundary_1)/sum(TEST_train_Fea_Y_1 == 2)]);
        train_acc_mdls(2,valcount) = mean([sum(predictions_tr_2(TEST_train_Fea_Y_23==1) < boundary_2)/sum(TEST_train_Fea_Y_23 == 1),...
                                     sum(predictions_tr_2(TEST_train_Fea_Y_23==2) >= boundary_2)/sum(TEST_train_Fea_Y_23 == 2)]);
        train_acc_mdls(3,valcount) = mean([sum(predictions_tr_3(TEST_train_Fea_Y_24==1)  < boundary_3)/sum(TEST_train_Fea_Y_24 == 1),...
                                     sum(predictions_tr_3(TEST_train_Fea_Y_24==2) >= boundary_3)/sum(TEST_train_Fea_Y_24 == 2)]);
        train_acc_mdls(4,valcount) = mean([sum(predictions_tr_4(TEST_train_Fea_Y_34==1)  < boundary_4)/sum(TEST_train_Fea_Y_34 == 1),...
                                     sum(predictions_tr_4(TEST_train_Fea_Y_34==2) >= boundary_4)/sum(TEST_train_Fea_Y_34 == 2)]); 

        isintest_init = (TEST_test_Fea_all{1}(:,1) ~= 0);
        isintest_Idle = isintest_init(1:3);
        isintest_Walk = isintest_init(4:6);
        isintest_Ascend = isintest_init(7:9);
        isintest_Descend = isintest_init(10:12);

        test_Fea_isIdle = repmat(TEST_test_Fea_all{1}(1:3,:),3,1);
        test_Fea_notIdle = TEST_test_Fea_all{1}(4:12,:);
        index_test_1 = repmat(isintest_Idle,3,1) & [isintest_Walk;...
                                              isintest_Ascend;...
                                              isintest_Descend];
        TEST_test_Fea_1 = [test_Fea_isIdle(index_test_1,:);test_Fea_notIdle(index_test_1,:)];
        TEST_test_Fea_Y_1 = [ones(sum(index_test_1),1);ones(sum(index_test_1),1)*2];

        test_Fea_23_tmp = TEST_test_Fea_all{2}(4:9,:);
        index_test_2 = isintest_Walk & isintest_Ascend;
        TEST_test_Fea_2 = test_Fea_23_tmp([index_test_2;index_test_2],:);
        TEST_test_Fea_Y_2 = [ones(sum(index_test_2),1);ones(sum(index_test_2),1)*2];

        test_Fea_24_tmp = TEST_test_Fea_all{3}([4:6,10:12],:);
        index_test_3 = isintest_Walk & isintest_Descend;
        TEST_test_Fea_3 = test_Fea_24_tmp([index_test_3;index_test_3],:);
        TEST_test_Fea_Y_3 = [ones(sum(index_test_3),1);ones(sum(index_test_3),1)*2];

        test_Fea_34_tmp = TEST_test_Fea_all{4}(7:12,:);
        index_test_4 = isintest_Ascend & isintest_Descend;
        TEST_test_Fea_4 = test_Fea_34_tmp([index_test_4;index_test_4],:);
        TEST_test_Fea_Y_4 = [ones(sum(index_test_4),1);ones(sum(index_test_4),1)*2];

        predictions_te_1 = predict(model_1, TEST_test_Fea_1);
        predictions_te_2 = predict(model_23, TEST_test_Fea_2);
        predictions_te_3 = predict(model_24, TEST_test_Fea_3);
        predictions_te_4 = predict(model_34, TEST_test_Fea_4);
                                 
        test_acc_mdls(1,valcount) = mean([sum(predictions_te_1(TEST_test_Fea_Y_1==1) < boundary_1)/sum(TEST_test_Fea_Y_1 == 1),...
                                     sum(predictions_te_1(TEST_test_Fea_Y_1==2) >= boundary_1)/sum(TEST_test_Fea_Y_1 == 2)]);
        test_acc_mdls(2,valcount) = mean([sum(predictions_te_2(TEST_test_Fea_Y_2==1) < boundary_2)/sum(TEST_test_Fea_Y_2 == 1),...
                                     sum(predictions_te_2(TEST_test_Fea_Y_2==2) >= boundary_2)/sum(TEST_test_Fea_Y_2 == 2)]);
        test_acc_mdls(3,valcount) = mean([sum(predictions_te_3(TEST_test_Fea_Y_3==1) < boundary_3)/sum(TEST_test_Fea_Y_3 == 1),...
                                     sum(predictions_te_3(TEST_test_Fea_Y_3==2) >= boundary_3)/sum(TEST_test_Fea_Y_3 == 2)]);
        test_acc_mdls(4,valcount) = mean([sum(predictions_te_4(TEST_test_Fea_Y_4==1) < boundary_4)/sum(TEST_test_Fea_Y_4 == 1),...
                                     sum(predictions_te_4(TEST_test_Fea_Y_4==2) >= boundary_4)/sum(TEST_test_Fea_Y_4 == 2)]); 

        predict_label_tmp = swlrPredict_4Class_ov3ovo(TEST_test_Fea_all,TEST_test_Fea_Y,swlrModels);
        predict_label_test = predict_label_tmp(isintest_init);
        TEST_test_Fea_Y_test = TEST_test_Fea_Y(isintest_init);

        test_acc_all(valcount) = sum(predict_label_test == TEST_test_Fea_Y_test)/size(TEST_test_Fea_Y_test,1);
        test_recall_4classes(1,valcount) = sum(predict_label_test(TEST_test_Fea_Y_test==1,1) == 1)/sum(TEST_test_Fea_Y_test == 1);
        test_recall_4classes(2,valcount) = sum(predict_label_test(TEST_test_Fea_Y_test==2,1) == 2)/sum(TEST_test_Fea_Y_test == 2);
        test_recall_4classes(3,valcount) = sum(predict_label_test(TEST_test_Fea_Y_test==3,1) == 3)/sum(TEST_test_Fea_Y_test == 3);
        test_recall_4classes(4,valcount) = sum(predict_label_test(TEST_test_Fea_Y_test==4,1) == 4)/sum(TEST_test_Fea_Y_test == 4);   

        test_precision_4classes(1,valcount) = sum(TEST_test_Fea_Y_test(predict_label_test==1,1) == 1)/sum(predict_label_test == 1);
        test_precision_4classes(2,valcount) = sum(TEST_test_Fea_Y_test(predict_label_test==2,1) == 2)/sum(predict_label_test == 2);
        test_precision_4classes(3,valcount) = sum(TEST_test_Fea_Y_test(predict_label_test==3,1) == 3)/sum(predict_label_test == 3);
        test_precision_4classes(4,valcount) = sum(TEST_test_Fea_Y_test(predict_label_test==4,1) == 4)/sum(predict_label_test == 4);   

    end

    test_acc_mdls(:,valfoldnum+1) = nanmean(test_acc_mdls(:,1:valfoldnum),2);
    train_acc_mdls(:,valfoldnum+1) = nanmean(train_acc_mdls(:,1:valfoldnum),2);

    test_acc_all(valfoldnum+1) = mean(test_acc_all(1:valfoldnum));
    test_recall_4classes(:,valfoldnum+1) = nanmean(test_recall_4classes(:,1:valfoldnum),2);
    test_precision_4classes(:,valfoldnum+1) = nanmean(test_precision_4classes(:,1:valfoldnum),2);
    test_precision_all(valfoldnum+1) = nanmean(test_precision_4classes(:,valfoldnum+1));
    test_F1_score_4classes(:,valfoldnum+1) = 2./(1./test_recall_4classes(:,valfoldnum+1) + 1./test_precision_4classes(:,valfoldnum+1));
    test_F1_score_all(valfoldnum+1) = 2./(1./test_acc_all(valfoldnum+1) + 1./test_precision_all(valfoldnum+1));
    
    test_acc_mdls_sort = sort(test_acc_mdls(:,1:valfoldnum),2,'descend');
    test_acc_mdls_max3 = mean(test_acc_mdls_sort(:,1:3),2);
    test_acc_all_sort = sort(test_acc_all(:,1:valfoldnum),2,'descend');
    test_acc_all_max3 = mean(test_acc_all_sort(:,1:3),2);
    
    test_all = [test_acc_mdls(:,valfoldnum+1);...
                test_acc_all(valfoldnum+1);...
                test_acc_mdls_max3;...
                test_acc_all_max3;...
                test_recall_4classes(:,valfoldnum+1);...
                test_precision_4classes(:,valfoldnum+1);...
                test_precision_all(valfoldnum+1);...
                test_F1_score_4classes(:,valfoldnum+1);...
                test_F1_score_all(valfoldnum+1)];

    save([offlinepath,'\result_',num2str(Subject(subjectnum).number), '_slr.mat'],...
        'train_acc_mdls','test_acc_mdls','test_acc_all','test_acc_mdls_max3','test_acc_all_max3',...
        'test_recall_4classes','test_precision_4classes','test_F1_score_4classes','test_all');

    
    fprintf('      acc_all   = %2.4f %2.4f %2.4f %2.4f %2.4f   average %.4f\n',test_acc_all);
    fprintf('      recall    = %2.4f %2.4f %2.4f %2.4f\n',test_recall_4classes(:,valfoldnum+1));
    fprintf('      precision = %2.4f %2.4f %2.4f %2.4f\n',test_precision_4classes(:,valfoldnum+1));
    fprintf('      F1_score  = %2.4f %2.4f %2.4f %2.4f\n',test_F1_score_4classes(:,valfoldnum+1));
    fprintf('      acc_mdls  = %2.4f %2.4f %2.4f %2.4f\n',test_acc_mdls(:,valfoldnum+1));
    
    ACC_train = [ACC_train;train_acc_mdls(:,valfoldnum+1)'];
    ACC_test = [ACC_test;test_all'];
    
    toc;
    
end

if size(ACC_train,1)>1
    ACC_train = [ACC_train;nanmean(ACC_train,1);nanstd(ACC_train)];
    ACC_test = [ACC_test;nanmean(ACC_test,1);nanstd(ACC_test)];
end

save('ACC_all_swlr.mat','ACC_test','ACC_train','Subject');
