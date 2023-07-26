function [acc,acc_train,score] = CGperformance_withoutfband(CG,mdltype)
%For one [c,g] pack in one model
%Give accuracy for one fold as its performance

[row,col]=size(CG);
if row>1 || col ~= 2
	error('输入的参数错误');
end
c_num = CG(1);
g_num = CG(2);

c_k = 0.3;
% cmd = ['-s 1 -t 2 ','-c ',num2str(2^c_num),' -g ',num2str(2^g_num),' -b 1'];
cmd = ['-c ',num2str(2^c_num),' -g ',num2str(2^g_num),' -b 1'];

files = dir('FeaSets_*.mat');
valfoldnum = length(files);

acc_2classtmp = zeros(1,valfoldnum);
acc_2class_traintmp = zeros(1,valfoldnum);
for valcount = 1:valfoldnum
    load(['FeaSets_',num2str(valcount),'_for_classifiers_trte.mat']);
    fea_num = size(VAL_train_Fea_1,2);
    fband_mdl = true(1,fea_num);
    
    isinval_init = (VAL_val_Fea_all{1}(:,1) ~= 0);
    isinval_Idle = isinval_init(1:3);
    isinval_Walk = isinval_init(4:6);
    isinval_Ascend = isinval_init(7:9);
    isinval_Descend = isinval_init(10:12);
    
    switch mdltype
        case 1
            train_Fea_Y = VAL_train_Fea_Y_1;
            train_Fea = VAL_train_Fea_1(:,fband_mdl);
            test_Fea_isIdle = repmat(VAL_val_Fea_all{1}(1:3,fband_mdl),3,1);
            test_Fea_notIdle = VAL_val_Fea_all{1}(4:12,fband_mdl);
            index_val = repmat(isinval_Idle,3,1) & [isinval_Walk;...
                                                  isinval_Ascend;...
                                                  isinval_Descend];
            test_Fea = [test_Fea_isIdle(index_val,:);test_Fea_notIdle(index_val,:)];
            test_Fea_Y = [ones(sum(index_val),1);ones(sum(index_val),1)*2];
        case 2
            train_Fea_Y = VAL_train_Fea_Y_23;
            train_Fea = VAL_train_Fea_23(:,fband_mdl);
            test_Fea_tmp = VAL_val_Fea_all{2}(4:9,fband_mdl);
            index_val = isinval_Walk & isinval_Ascend;
            test_Fea = test_Fea_tmp([index_val;index_val],:);
            test_Fea_Y = [ones(sum(index_val),1);ones(sum(index_val),1)*2];
        case 3
            train_Fea_Y = VAL_train_Fea_Y_24;
            train_Fea = VAL_train_Fea_24(:,fband_mdl);
            test_Fea_tmp = VAL_val_Fea_all{3}([4:6,10:12],fband_mdl);
            index_val = isinval_Walk & isinval_Descend;
            test_Fea = test_Fea_tmp([index_val;index_val],:);
            test_Fea_Y = [ones(sum(index_val),1);ones(sum(index_val),1)*2];
        case 4
            train_Fea_Y = VAL_train_Fea_Y_34;
            train_Fea = VAL_train_Fea_34(:,fband_mdl);
            test_Fea_tmp = VAL_val_Fea_all{4}(7:12,fband_mdl);
            index_val = isinval_Ascend & isinval_Descend;
            test_Fea = test_Fea_tmp([index_val;index_val],:);
            test_Fea_Y = [ones(sum(index_val),1);ones(sum(index_val),1)*2];
        otherwise
            error('无法识别输入的二分类模型类别');
    end

    model = svmtrain(train_Fea_Y,train_Fea,cmd);
    [predict_label_te,accuracy_te,prob_estimates_te] = svmpredict(test_Fea_Y,test_Fea,model,' -b 1');
    acc_2classtmp(valcount) = accuracy_te(1);
    [predict_label_tr,accuracy_tr,prob_estimates_tr] = svmpredict(train_Fea_Y,train_Fea,model,' -b 1');
    acc_2class_traintmp(valcount) = accuracy_tr(1);

end

acc = nanmean(acc_2classtmp);
acc_train = mean(acc_2class_traintmp);
score = acc - c_k * abs(acc_train-acc);

fmt = repmat('%.4f  ',1,valfoldnum);
fprintf(['ACC = ',fmt,'average %.4f\n'], acc_2classtmp, acc);
fprintf(['ACC_train = ',fmt,'average %.4f\n\n'], acc_2class_traintmp, acc_train);
fprintf('score = %.4f\n\n', score);




