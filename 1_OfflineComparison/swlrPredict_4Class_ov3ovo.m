function PredictType = swlrPredict_4Class_ov3ovo(test_Fea,Y,slwrModels)
% input: test_Fea: test_Fea generated by the Y-th csp.

    %% modify input feature samples    
    assert(size(test_Fea{1},1) == size(test_Fea{2},1) && size(test_Fea{2},1) == size(test_Fea{3},1)...
    && size(test_Fea{3},1) == size(test_Fea{4},1),'Predit function <svmPredict_4Class_ov3ovo> input error!!');

    testsamplenum = size(test_Fea{1},1);
    
    predictions_1 = predict(slwrModels.Mdl_1, test_Fea{1});
    predictions_2 = predict(slwrModels.Mdl_23, test_Fea{2});
    predictions_3 = predict(slwrModels.Mdl_24, test_Fea{3});
    predictions_4 = predict(slwrModels.Mdl_34, test_Fea{4});
    
    predict_label_te_1 = zeros(size(predictions_1));
    predict_label_te_2 = zeros(size(predictions_2));
    predict_label_te_3 = zeros(size(predictions_3));
    predict_label_te_4 = zeros(size(predictions_4));

    for i = 1:size(Y,1)
        if (predictions_1(i) < slwrModels.boundary_1)
            predict_label_te_1(i) = 1;
        else
            predict_label_te_1(i) = 2;
        end
        if (predictions_2(i) < slwrModels.boundary_2)
            predict_label_te_2(i) = 1;
        else
            predict_label_te_2(i) = 2;
        end
        if (predictions_3(i) < slwrModels.boundary_3)
            predict_label_te_3(i) = 1;
        else
            predict_label_te_3(i) = 2;
        end
        if (predictions_4(i) < slwrModels.boundary_4)
            predict_label_te_4(i) = 1;
        else
            predict_label_te_4(i) = 2;
        end
    end

    %%%%%%%%%%%%%%%
    
    vote = zeros(testsamplenum,4);
    prob = zeros(testsamplenum,4);
    predict_label_te = zeros(testsamplenum,1);
    for i = 1:testsamplenum
        if predict_label_te_1(i) == 1
            vote(i,:) = [1,0,0,0];
            predict_label_te(i) = 1;
        else
            if predict_label_te_2(i) == 1
                vote(i,2) = vote(i,2)+1;
                prob(i,2) = abs(predictions_2(i) - slwrModels.boundary_2);
            else
                vote(i,3) = vote(i,3)+1;
                prob(i,3) = abs(predictions_2(i) - slwrModels.boundary_2);
            end
            if predict_label_te_3(i) == 1
                vote(i,2) = vote(i,2)+1;
                prob(i,2) = abs(predictions_3(i) - slwrModels.boundary_3);
            else
                vote(i,4) = vote(i,4)+1;
                prob(i,4) = abs(predictions_3(i) - slwrModels.boundary_3);
            end
            if predict_label_te_4(i) == 1
                vote(i,3) = vote(i,3)+1;
                prob(i,3) = abs(predictions_4(i) - slwrModels.boundary_4);
            else
                vote(i,4) = vote(i,4)+1;
                prob(i,4) = abs(predictions_4(i) - slwrModels.boundary_4);
            end
            if max(vote(i,:)) == 2
                [~,predict_label_te(i)] = max(vote(i,:));
            else
                [~,predict_label_te(i)] = max(prob(i,:));
            end
        end
    end

    %% 6 predictions fusing    
    PredictType = predict_label_te;
end
    