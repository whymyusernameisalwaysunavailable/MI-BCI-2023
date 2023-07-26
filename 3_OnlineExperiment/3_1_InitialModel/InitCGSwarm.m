function [ParSwarm,OptSwarm]=InitCGSwarm(SwarmSize,ParticleSize,ParticleScope,...
                                        mdltype,...
                                        AdaptFunc)
%������������ʼ������Ⱥ���޶�����Ⱥ��λ���Լ��ٶ���ָ���ķ�Χ��
%[ParSwarm,OptSwarm,BadSwarm]=InitSwarm(SwarmSize,ParticleSize,ParticleScope,AdaptFunc)
%
%���������SwarmSize:��Ⱥ��С�ĸ���
%���������ParticleSize��һ�����ӵ�ά��
%���������ParticleScope:һ�������������и�ά�ķ�Χ��
%���������������� ParticleScope��ʽ:
%�������������������� 3ά���ӵ�ParticleScope��ʽ:
%�������������������������������������������������������������������� [x1Min,x1Max
%���������������������������������������������������������������������� x2Min,x2Max
%���������������������������������������������������������������������� x3Min,x3Max]
%
%���������AdaptFunc����Ӧ�Ⱥ���
%
%�����ParSwarm��ʼ��������Ⱥ
%�����OptSwarm����Ⱥ��ǰ���Ž���ȫ�����Ž�
%
%�÷�[ParSwarm,OptSwarm,BadSwarm]=InitSwarm(SwarmSize,ParticleSize,ParticleScope,AdaptFunc);
%
%�쳣�����ȱ�֤���ļ���Matlab������·���У�Ȼ��鿴��ص���ʾ��Ϣ��
%
%�����ˣ�XXX
%����ʱ�䣺2007.3.26
%�ο����ף���
%

%�ݴ����
if nargin~=5
        error('����Ĳ�����������')
end
if nargout<2
        error('����Ĳ����ĸ���̫�٣����ܱ�֤�Ժ�����С�');
end

[row,colum]=size(ParticleSize);
if row>1 || colum>1
        error('��������ӵ�ά��������һ��1��1�е����ݡ�');
end
[row,colum]=size(ParticleScope);
if row~=ParticleSize || colum~=2
        error('��������ӵ�ά����Χ����');
end

%��ʼ������Ⱥ����

%��ʼ������Ⱥ����ȫ����Ϊ[0-1]�����
%rand('state',0);
ParSwarm=rand(SwarmSize,2*ParticleSize+1);

%������Ⱥ��λ��,�ٶȵķ�Χ���е���
for k=1:ParticleSize
        ParSwarm(:,k)=ParSwarm(:,k)*(ParticleScope(k,2)-ParticleScope(k,1))+ParticleScope(k,1);
         %�����ٶȣ�ʹ�ٶ���λ�õķ�Χһ��
        ParSwarm(:,ParticleSize+k)=ParSwarm(:,ParticleSize+k)*(ParticleScope(k,2)-ParticleScope(k,1))+ParticleScope(k,1);
end
	 
%��ÿһ�����Ӽ�������Ӧ�Ⱥ�����ֵ

for k=1:SwarmSize
        ParSwarm(k,2*ParticleSize+1)=AdaptFunc(ParSwarm(k,1:ParticleSize),...
                                               mdltype);
end

%��ʼ������Ⱥ���Ž����
OptSwarm=zeros(SwarmSize+1,ParticleSize+1);
%����Ⱥ���Ž����ȫ����Ϊ��
[maxValue,row]=max(ParSwarm(:,2*ParticleSize+1));
%Ѱ����Ӧ�Ⱥ���ֵ���Ľ��ھ����е�λ��(����)
OptSwarm(1:SwarmSize,1:ParticleSize)=ParSwarm(1:SwarmSize,1:ParticleSize);
OptSwarm(1:SwarmSize,ParticleSize+1)=ParSwarm(1:SwarmSize,2*ParticleSize+1);
OptSwarm(SwarmSize+1,:)=OptSwarm(row,:);
