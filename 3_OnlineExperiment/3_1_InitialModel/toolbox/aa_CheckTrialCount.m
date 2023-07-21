TotalTrial = [];
for runtime = 1:10000
    TrialNum = 60;
    RandomTrial = randsrc(TrialNum,1,[2,3,4]);
    % 避免上-下或下-上台阶（动作不符合自然习惯）
    for TrialIndex = 1:TrialNum
        if TrialIndex==TrialNum  % 最后一个trial动作不做判定
            break;
        elseif RandomTrial(TrialIndex)==3 && RandomTrial(TrialIndex+1)==4 % 若上-下台阶，第二个动作换为随机步行/上台阶
            RandomTrial(TrialIndex+1) = randsrc(1,1,[2,3]);
        elseif RandomTrial(TrialIndex)==4 && RandomTrial(TrialIndex+1)==3 % 若下-上台阶，第二个动作换为随机步行/下台阶
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