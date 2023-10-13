%"C:/Users/jagad/Downloads/Nifty/niftydailydata.xlsx",'Sheet1','d2:ay1531'
%"C:/Users/jagad/Downloads/Nifty/niftydailydata.xlsx",'Sheet1','b2:b1531'
%"C:/Users/jagad/Downloads/Nifty/niftyfinalsimout63A.xlsx",'Sheet1','b2:aw1601'
%"C:/Users/jagad/Downloads/DAX 30/dax30dailydata.xlsx",'Sheet2','b2:b1565'
%"C:/Users/jagad/Downloads/DAX 30/dax30simout.xlsx",'Sheet1','b2:aa1601'
%"C:/Users/jagad/Downloads/euro 50/euro50.xlsx",'Sheet2','d2:ax1839'
%"C:/Users/jagad/Downloads/euro 50/euro50.xlsx",'Sheet2','b2:b1839'
%"C:/Users/jagad/Downloads/euro 50/euro50finalsimout.xlsx",'Sheet1','b2:av1601'
%"C:/Users/jagad/Downloads/Australia/Australia ASX 50 Daily Data (1).xlsx",'Sheet2','d2:AW1568'
%"C:/Users/jagad/Downloads/Australia/Australia ASX 50 Daily Data (1).xlsx",'Sheet2','b2:b1568'
%"C:/Users/jagad/Downloads/Australia/Australiafinalsimulation.xlsx",'Sheet1','b2:AU1601'


format short
%n=5; %Number of assets
m=100; %No. of simulations
returnsrolling=xlsread("C:/Users/jagad/Downloads/Euro Phases/Neutral_Phase.xlsx",'Sheet2','d2:AX751');
market=xlsread("C:/Users/jagad/Downloads/Euro Phases/Neutral_Phase.xlsx",'Sheet2', 'b2:b751');
[h,n]=size(returnsrolling);
mu0=mean(market);
mu1=0.2;
tradeoff=1;
%returns=zeros(T,n);
%meanreturns=mean(returns);
%stdreturns=std(returns);
% a=meanreturns-stdreturns;
% b=meanreturns+stdreturns;
% X=zeros(m,n);
% for hh=1:n
% X(:,hh)=randn(m,1)*(b(hh)-a(hh))+a(hh);
% end
Xrolling=xlsread("C:/Users/jagad/Downloads/Euro Phases/Euro_Neutral_FinalsOut.xlsx",'Sheet1','b2:AV101');
%[Xwh, mu, invMat, whMat] = whiten(X,0.0001);
%X=rand(m,n);
T=500;
t1=250;
svminsample=0;
svmoutsample=0;
bertoutsample=0;
nonroutsample=0;
for oo=1:1
returns=returnsrolling((oo-1)*t1+1:(oo-1)*t1+T,1:end);
X=Xrolling((oo-1)*m+1:oo*m,1:end);
outreturns=returnsrolling((oo-1)*t1+T+1:(oo-1)*t1+T+t1,1:end);
sigm=cov(X);
%%sigm2=round(sigm,4);
%invsigm=inv(sigm);
%whMat=chol(invsigm)'; % transpose of cholesky decomposition of cov(inverse)
[Xwh, mu, invMat, whMat] = whiten(X,0.0001);
%whMat=eye(n);
l=zeros(1,n);
for k=1:n
    l(k)=max(whMat(:,k)'*X.')-min(whMat(:,k)'*X.')+0.000001;
end
suml=sum(l);
K=zeros(m,m);
for i=1:m
 for j=1:m
     K(i,j)=suml-sum(abs(whMat*(X(i,:)'-X(j,:)')));
 end
end
K1=zeros(m,1);
for r=1:m
    K1(r)=K(r,r);
end
cvx_begin
variable alpha1(m)
variable A
minimize A
%minimize alpha1'*K*alpha1
subject to
alpha1'*K*alpha1-alpha1'*K1<=A;
alpha1<=(1/(m*mu1));
sum(alpha1)==1;
alpha1>=0;
cvx_end
alpha2=round(alpha1,4);
supvecindex=find(alpha2>0); %support vector index
dim=numel(supvecindex);
U=zeros(dim,n);
for kl=1:dim
    U(kl,:)=X(supvecindex(kl),:);  %Support vectors
end
supalpha=zeros(dim,1);
for jj=1:dim
    supalpha(jj)= alpha2(supvecindex(jj)); %support alpha
end
bounsupvecindex=find(alpha2>0 & alpha2<round(1/(m*mu1),4));
bdim=numel(bounsupvecindex);
bU=zeros(bdim,n);
for kk=1:bdim
    bU(kk,:)=X(bounsupvecindex(kk),:);  %boundary Support vectors
end
theta1=zeros(bdim,1);
 for ii=1:bdim
     for hh=1:dim
    theta1(ii)=theta1(ii)+supalpha(hh)*sum(abs(whMat*(bU(ii,:)'-U(hh,:)')));
     end
 end
thetafinal=min(theta1);
Wh=whMat*U';
cvx_begin
%cvx_solver mosek
variable u(T)
variable x(n)
variable etaA
variable muA(dim,n)
variable tauA(dim)
variable lamdaA(dim,n)
variable muB(dim,n)
variable tauB(dim)
variable lamdaB(dim,n)
variable etaB
variable delta1
variable sumA(dim)
variable sumB(dim)
variable B
%variable z(n) binary
minimize B
subject to:
sum(u)/T+tradeoff*(sum(sum((muA-lamdaA)'.*Wh))+thetafinal*etaA)<=B;
whMat*(sum(lamdaA-muA))'-x==0;
 for nn=1:dim
       muA(nn,:)+lamdaA(nn,:)==etaA*supalpha(nn)*ones(1,n);
    end
% for mm=1:dim
%     tauA(mm)==(muA(mm,:)-lamdaA(mm,:))*whMat*U(mm,:)';
%      sumA(mm)==whMat*(lamdaA(mm,:)-muA(mm,:))';
% end
%whMat*(sum(lamdaA-muA))'+x==0;
%u+returns*x>=sum(sum((muA-lamdaA)'.*Wh))+thetafinal*etaA;
    %u+returns*x >= sum(tauA)+thetafinal*etaA;
  %sum(sumA)+x==0;
%     for nn=1:dim
%        muA(nn,:)+lamdaA(nn,:)==etaA*supalpha(nn)*ones(1,n);
%     end
    
%     for mm=1:dim
%     tauB(mm)==(muB(mm,:)-lamdaB(mm,:))*whMat*U(mm,:)';
%     sumB(mm)==whMat*(lamdaB(mm,:)-muB(mm,:))';
%     end
    %sum(tauB)+theta1*etaB >=-0.02;
    %sum(sumB)+x==0; 
         for ku=1:dim
        muB(ku,:)+lamdaB(ku,:)==etaB*supalpha(ku)*ones(1,n);
     end
     whMat*(sum(lamdaB-muB))'+x==0;
     sum(sum((muB-lamdaB)'.*Wh))+thetafinal*etaB-returns*x<=u;
    for pp=1:dim
        for pp1=1:n
           muA(pp,pp1)>=0;
           lamdaA(pp,pp1)>=0;
          muB(pp,pp1)>=0;
           lamdaB(pp,pp1)>=0;
        end
    end
    sum(x)==1;
    x>=0;
    %x<=z;
    %sum(z)==10;
    etaA>=0;
    u>=0;
    etaB>=0;     
cvx_end
svmout=outreturns*x;
svmin=returns*x;
svminsample=[svminsample;svmin]
svmoutsample=[svmoutsample;svmout];

%COMPARISON MODELS
bertreturns=sum(returns)/T;
bertgamma=12.299884705508642;
bertrhat=0.003;
%ROBUST SEMI MAD BERTSIMAS AND SIM
cvx_begin
variable bertA
variable bertw(n)
variable bertu(T)
variable bertx(n)
variable bertp(n)
variable bertz0
minimize bertA
subject to:
1/(T*tradeoff)*sum(bertu)-(bertA/tradeoff)-bertreturns*bertx+sum(bertp)+bertgamma*bertz0<=0;
-returns*bertx-bertu+bertreturns*bertx+sum(bertp)+bertgamma*bertz0<=0;
for ib=1:n
    bertz0+bertp(ib)>=bertrhat*bertw(ib);
end
sum(bertx)==1;
bertx<=bertw;
-bertw<=bertx;
bertw>=0;
bertp>=0;
bertz0>=0;
bertu>=0;
bertx>=0;
cvx_end
bertout=outreturns*bertx;
bertoutsample=[bertoutsample;bertout];


%ASSYMETRY ROBUST MAD
% cvx_begin
% variable assyA
% variable assyu(T)
% variable assyx(n)
% variable assyh
% variable assyr(n)
% variable assys(n)
% variable assyd(T)
% variable assyg(n)
% variable assy
%cvx_end
%SEMIMAD NOMINAL
cvx_begin
variable v(T)
variable y(n)
minimize sum(v)/T-tradeoff*bertreturns*y
subject to:
v>=bertreturns*y-returns*y;
v>=0;
sum(y)==1;
y>=0;
cvx_end
nonrout=outreturns*y;
nonroutsample=[nonroutsample;nonrout];
end
%xlswrite('robustsvcdowjonesout.xlsx',svminsample,'svmoutsamplemu0.2')
 xlswrite('robustsemimadEURO_NEUTRALPhasesout02.xlsx', svmoutsample,'svmoutsample')
 xlswrite('robustsemimadEURO_NEUTRALPhasesout02.xlsx', bertoutsample,'bertoutsample')
 xlswrite('robustsemimadEURO_NEUTRALPhasesout02.xlsx', nonroutsample,'nonroutsample')