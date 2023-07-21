TotalTrial = [];
for runtime = 1:10000
    TrialNum = 60;
    RandomTrial = randsrc(TrialNum,1,[2,3,4]);
    % ������-�»���-��̨�ף�������������Ȼϰ�ߣ�
    for TrialIndex = 1:TrialNum
        if TrialIndex==TrialNum  % ���һ��trial���������ж�
            break;
        elseif RandomTrial(TrialIndex)==3 && RandomTrial(TrialIndex+1)==4 % ����-��̨�ף��ڶ���������Ϊ�������/��̨��
            RandomTrial(TrialIndex+1) = randsrc(1,1,[2,3]);
        elseif RandomTrial(TrialIndex)==4 && RandomTrial(TrialIndex+1)==3 % ����-��̨�ף��ڶ���������Ϊ�������/��̨��
            RandomTrial(TrialIndex+1) = randsrc(1,1,[2,4]);
        end 
    end
    TotalTrial = [TotalTrial;RandomTrial];
end
WalkProb = sum(TotalTrial(:)==2) / size(TotalTrial,1);
UpProb = sum(TotalTrial(:)==3) / size(TotalTrial,1);
DownProb = sum(TotalTrial(:)==4) / size(TotalTrial,1);
totalProb = [WalkProb;UpProb;DownProb];
plot(totalProb,'+')
set(gca,'XTick',[0:1:5])