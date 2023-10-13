format long
returnsrolling=xlsread("C:\Users\jagad\Downloads\Euro Phases\eurostoxx 50 daily Data (1).xlsx",'UP','d2:AZ264');
%market=xlsread('dowjonesusdaily.xlsx','Sheet2','b502:b1558');
[h,n]=size(returnsrolling);
y=zeros(h,1);
w=ones(n,1)*1/n;
for t=1:h
    y(t)=returnsrolling(t,:)*w;
end
xlswrite('C:\Users\jagad\Downloads\equalweight.xlsx',y,'UP_Phase_EURO_outsample')
