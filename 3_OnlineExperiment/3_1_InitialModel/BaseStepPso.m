function [ParSwarm,OptSwarm,num_stop]=BaseStepPso(ParSwarm,OptSwarm,...
                                         mdltype,...
                                         AdaptFunc,ParticleScope,MaxW,MinW,LoopCount,CurCount,num_stop)
%功能描述：全局版本：基本的粒子群算法的单步更新位置,速度的算法
%
%[ParSwarm,OptSwarm]=BaseStepPso(ParSwarm,OptSwarm,AdaptFunc,ParticleScope,MaxW,MinW,LoopCount,CurCount)
%
%输入参数：ParSwarm:粒子群矩阵，包含粒子的位置，速度与当前的目标函数值
%输入参数：OptSwarm：包含粒子群个体最优解与全局最优解的矩阵
%输入参数：ParticleScope:一个粒子在运算中各维的范围；
%输入参数：AdaptFunc：适应度函数
%输入参数：LoopCount：迭代的总次数
%输入参数：CurCount：当前迭代的次数
%
%返回值：含意同输入的同名参数
%
%用法：[ParSwarm,OptSwarm]=BaseStepPso(ParSwarm,OptSwarm,AdaptFunc,ParticleScope,MaxW,MinW,LoopCount,CurCount)
%
%异常：首先保证该文件在Matlab的搜索路径中，然后查看相关的提示信息。
%
%编制人：XXX
%编制时间：2007.3.26
%参考文献：XXX
%参考文献：XXX
%
%修改记录
%----------------------------------------------------------------
%2007.3.27
%修改人：XXX
% 添加2*unifrnd(0,1).*SubTract1(row,:)中的unifrnd(0,1)随机数，使性能大为提高
%参照基于MATLAB的粒子群优化算法程序设计
%
% 总体评价：使用这个版本的调节系数，效果比较好
%

%容错控制
if nargin~=10
	error('输入的参数个数错误。')
end
if nargout~=3
	error('输出的个数太少，不能保证循环迭代。')
end

%开始单步更新的操作

%*********************************************
%*****更改下面的代码，可以更改惯性因子的变化*****
%---------------------------------------------------------------------
%线形递减策略
w=MaxW-CurCount*((MaxW-MinW)/LoopCount);
%---------------------------------------------------------------------
%w固定不变策略
%w=0.7;
%---------------------------------------------------------------------
%参考文献：陈贵敏，贾建援，韩琪，粒子群优化算法的惯性权值递减策略研究，西安交通大学学报，2006，1
%w非线形递减，以凹函数递减
%w=(MaxW-MinW)*(CurCount/LoopCount)^2+(MinW-MaxW)*(2*CurCount/LoopCount)+MaxW;
%---------------------------------------------------------------------
%w非线形递减，以凹函数递减
%w=MinW*(MaxW/MinW)^(1/(1+10*CurCount/LoopCount));
%*****更改上面的代码，可以更改惯性因子的变化*****
%*********************************************

%得到粒子群群体大小以及一个粒子维数的信息
[ParRow,ParCol]=size(ParSwarm);
%得到粒子的维数
ParCol=(ParCol-1)/2;
SubTract1=OptSwarm(1:ParRow,1:ParCol)-ParSwarm(:,1:ParCol);

%*********************************************
%*****更改下面的代码，可以更改c1,c2的变化*****
% c1=2;
% c2=2;
% c1=0.75;c2=0.75;
c1=0.35;c2=0.35;
%---------------------------------------------------------------------
%con=1;
%c1=4-exp(-con*abs(mean(ParSwarm(:,2*ParCol+1))-AdaptFunc(OptSwarm(ParRow+1,:))));
%c2=4-c1;
%----------------------------------------------------------------------
%*****更改上面的代码，可以更改c1,c2的变化*****
%*********************************************
b=0;
for row=1:ParRow
% 	SubTract2=OptSwarm(ParRow+1,:)-ParSwarm(row,1:ParCol);
    SubTract2=OptSwarm(ParRow+1,1:ParCol)-ParSwarm(row,1:ParCol);
% 	TempV=w.*ParSwarm(row,ParCol+1:2*ParCol)+c1*unifrnd(0,1).*SubTract1(row,:)+c2*unifrnd(0,1).*SubTract2;
    TempV=c1*unifrnd(0,1).*SubTract1(row,:)+c2*unifrnd(0,1).*SubTract2;
	%限制速度的代码
	for h=1:ParCol
            if TempV(:,h)>ParticleScope(h,2)
                TempV(:,h)=ParticleScope(h,2);
            end
            if TempV(:,h)<-ParticleScope(h,2)
                TempV(:,h)=-ParticleScope(h,2)+1e-10; %加1e-10防止适应度函数被零除
            end
    end 
	 
	%更新速度
	ParSwarm(row,ParCol+1:2*ParCol)=TempV;
	
	%*********************************************
	%*****更改下面的代码，可以更改约束因子的变化*****
	%---------------------------------------------------------------------
	%a=1;
	%---------------------------------------------------------------------
% 	a=0.729;
    a=1;
	 %*****更改上面的代码，可以更改约束因子的变化*****
	%*********************************************
	 
	 %限制位置的范围
	TempPos=ParSwarm(row,1:ParCol)+a*TempV;
	for h=1:ParCol
            if TempPos(:,h)>=ParticleScope(h,2)
                TempPos(:,h)=ParticleScope(h,2)-1e-10;
            end
            if TempPos(:,h)<=ParticleScope(h,1)
                TempPos(:,h)=ParticleScope(h,1)+1e-10;	
            end
    end

	%更新位置 
	ParSwarm(row,1:ParCol)=TempPos;
	 
	%计算每个粒子的新的适应度值
	ParSwarm(row,2*ParCol+1) = AdaptFunc(ParSwarm(row,1:ParCol),...
                                       mdltype);
    if ParSwarm(row,2*ParCol+1) > OptSwarm(row,ParCol+1)
        OptSwarm(row,1:ParCol)=ParSwarm(row,1:ParCol);
        OptSwarm(row,ParCol+1)=ParSwarm(row,2*ParCol+1);
    end
    if OptSwarm(row,ParCol+1)>OptSwarm(ParRow+1,ParCol+1)
        OptSwarm(ParRow+1,:)=OptSwarm(row,:);
        SwarmSize = ParRow;
        ParticleSize = ParCol;
%         ParSwarm(row,:)=rand(1,2*ParticleSize+1);
        ParSwarm=rand(SwarmSize,2*ParticleSize+1);
        % adjust the location and velocity in particle swarm
        for k=1:ParticleSize
                ParSwarm(:,k)=ParSwarm(:,k)*(ParticleScope(k,2)-ParticleScope(k,1))+ParticleScope(k,1);
                  % adjust velocity, make the velocity and position correspond to the same range
                ParSwarm(:,ParticleSize+k)=ParSwarm(:,ParticleSize+k)*(ParticleScope(k,2)-ParticleScope(k,1))+ParticleScope(k,1);
        end        
        b=1;
    end
end
    if b==0
       num_stop=num_stop+1;
    else
       num_stop=0;
    end
%for循环结束

%寻找适应度函数值最大的解在矩阵中的位置(行数)，进行全局最优的改变 
% [maxValue,row]=max(ParSwarm(:,2*ParCol+1));
% accParSwarm = AdaptFunc(ParSwarm(row,1:ParCol),...
%                        mdltype);
% accOptSwarm = AdaptFunc(OptSwarm(ParRow+1,1:ParCol),...
%                        mdltype);
% if accParSwarm > accOptSwarm
% 	 OptSwarm(ParRow+1,:)=ParSwarm(row,1:ParCol);
% end
