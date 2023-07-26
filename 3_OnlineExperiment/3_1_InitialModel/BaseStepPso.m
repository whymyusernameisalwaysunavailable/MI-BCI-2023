function [ParSwarm,OptSwarm,num_stop]=BaseStepPso(ParSwarm,OptSwarm,...
                                         mdltype,...
                                         AdaptFunc,ParticleScope,MaxW,MinW,LoopCount,CurCount,num_stop)
%����������ȫ�ְ汾������������Ⱥ�㷨�ĵ�������λ��,�ٶȵ��㷨
%
%[ParSwarm,OptSwarm]=BaseStepPso(ParSwarm,OptSwarm,AdaptFunc,ParticleScope,MaxW,MinW,LoopCount,CurCount)
%
%���������ParSwarm:����Ⱥ���󣬰������ӵ�λ�ã��ٶ��뵱ǰ��Ŀ�꺯��ֵ
%���������OptSwarm����������Ⱥ�������Ž���ȫ�����Ž�ľ���
%���������ParticleScope:һ�������������и�ά�ķ�Χ��
%���������AdaptFunc����Ӧ�Ⱥ���
%���������LoopCount���������ܴ���
%���������CurCount����ǰ�����Ĵ���
%
%����ֵ������ͬ�����ͬ������
%
%�÷���[ParSwarm,OptSwarm]=BaseStepPso(ParSwarm,OptSwarm,AdaptFunc,ParticleScope,MaxW,MinW,LoopCount,CurCount)
%
%�쳣�����ȱ�֤���ļ���Matlab������·���У�Ȼ��鿴��ص���ʾ��Ϣ��
%
%�����ˣ�XXX
%����ʱ�䣺2007.3.26
%�ο����ף�XXX
%�ο����ף�XXX
%
%�޸ļ�¼
%----------------------------------------------------------------
%2007.3.27
%�޸��ˣ�XXX
% ���2*unifrnd(0,1).*SubTract1(row,:)�е�unifrnd(0,1)�������ʹ���ܴ�Ϊ���
%���ջ���MATLAB������Ⱥ�Ż��㷨�������
%
% �������ۣ�ʹ������汾�ĵ���ϵ����Ч���ȽϺ�
%

%�ݴ����
if nargin~=10
	error('����Ĳ�����������')
end
if nargout~=3
	error('����ĸ���̫�٣����ܱ�֤ѭ��������')
end

%��ʼ�������µĲ���

%*********************************************
%*****��������Ĵ��룬���Ը��Ĺ������ӵı仯*****
%---------------------------------------------------------------------
%���εݼ�����
w=MaxW-CurCount*((MaxW-MinW)/LoopCount);
%---------------------------------------------------------------------
%w�̶��������
%w=0.7;
%---------------------------------------------------------------------
%�ο����ף��¹������ֽ�Ԯ������������Ⱥ�Ż��㷨�Ĺ���Ȩֵ�ݼ������о���������ͨ��ѧѧ����2006��1
%w�����εݼ����԰������ݼ�
%w=(MaxW-MinW)*(CurCount/LoopCount)^2+(MinW-MaxW)*(2*CurCount/LoopCount)+MaxW;
%---------------------------------------------------------------------
%w�����εݼ����԰������ݼ�
%w=MinW*(MaxW/MinW)^(1/(1+10*CurCount/LoopCount));
%*****��������Ĵ��룬���Ը��Ĺ������ӵı仯*****
%*********************************************

%�õ�����ȺȺ���С�Լ�һ������ά������Ϣ
[ParRow,ParCol]=size(ParSwarm);
%�õ����ӵ�ά��
ParCol=(ParCol-1)/2;
SubTract1=OptSwarm(1:ParRow,1:ParCol)-ParSwarm(:,1:ParCol);

%*********************************************
%*****��������Ĵ��룬���Ը���c1,c2�ı仯*****
% c1=2;
% c2=2;
% c1=0.75;c2=0.75;
c1=0.35;c2=0.35;
%---------------------------------------------------------------------
%con=1;
%c1=4-exp(-con*abs(mean(ParSwarm(:,2*ParCol+1))-AdaptFunc(OptSwarm(ParRow+1,:))));
%c2=4-c1;
%----------------------------------------------------------------------
%*****��������Ĵ��룬���Ը���c1,c2�ı仯*****
%*********************************************
b=0;
for row=1:ParRow
% 	SubTract2=OptSwarm(ParRow+1,:)-ParSwarm(row,1:ParCol);
    SubTract2=OptSwarm(ParRow+1,1:ParCol)-ParSwarm(row,1:ParCol);
% 	TempV=w.*ParSwarm(row,ParCol+1:2*ParCol)+c1*unifrnd(0,1).*SubTract1(row,:)+c2*unifrnd(0,1).*SubTract2;
    TempV=c1*unifrnd(0,1).*SubTract1(row,:)+c2*unifrnd(0,1).*SubTract2;
	%�����ٶȵĴ���
	for h=1:ParCol
            if TempV(:,h)>ParticleScope(h,2)
                TempV(:,h)=ParticleScope(h,2);
            end
            if TempV(:,h)<-ParticleScope(h,2)
                TempV(:,h)=-ParticleScope(h,2)+1e-10; %��1e-10��ֹ��Ӧ�Ⱥ��������
            end
    end 
	 
	%�����ٶ�
	ParSwarm(row,ParCol+1:2*ParCol)=TempV;
	
	%*********************************************
	%*****��������Ĵ��룬���Ը���Լ�����ӵı仯*****
	%---------------------------------------------------------------------
	%a=1;
	%---------------------------------------------------------------------
% 	a=0.729;
    a=1;
	 %*****��������Ĵ��룬���Ը���Լ�����ӵı仯*****
	%*********************************************
	 
	 %����λ�õķ�Χ
	TempPos=ParSwarm(row,1:ParCol)+a*TempV;
	for h=1:ParCol
            if TempPos(:,h)>=ParticleScope(h,2)
                TempPos(:,h)=ParticleScope(h,2)-1e-10;
            end
            if TempPos(:,h)<=ParticleScope(h,1)
                TempPos(:,h)=ParticleScope(h,1)+1e-10;	
            end
    end

	%����λ�� 
	ParSwarm(row,1:ParCol)=TempPos;
	 
	%����ÿ�����ӵ��µ���Ӧ��ֵ
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
%forѭ������

%Ѱ����Ӧ�Ⱥ���ֵ���Ľ��ھ����е�λ��(����)������ȫ�����ŵĸı� 
% [maxValue,row]=max(ParSwarm(:,2*ParCol+1));
% accParSwarm = AdaptFunc(ParSwarm(row,1:ParCol),...
%                        mdltype);
% accOptSwarm = AdaptFunc(OptSwarm(ParRow+1,1:ParCol),...
%                        mdltype);
% if accParSwarm > accOptSwarm
% 	 OptSwarm(ParRow+1,:)=ParSwarm(row,1:ParCol);
% end
