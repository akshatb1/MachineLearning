%GMM Based Segmentation

clc
clear all
image= im2double(imread('ski_image.jpg'));
image= imresize(image,0.5);
[a b c]= size(image);
feature=cell(1,a*b);
mu=cell(1,3);
mu{1}=[.47 .47 .47]';
mu{2}=[0.05 0.05 0.05]';
mu{3}=[0.7 0.7 0.7]';
cov=cell(1,3);
pii=cell(1,3);
for k=1:3
    cov{k}=eye(3);
    pii{k}=1/3;
end

k=1;
for i=1:a
    for j=1:b
        temp_feature{k}=image(i,j,:);
        k=k+1;  
    end
end
for p=1:a*b
    for i=1:3
        feature{p}(i)=temp_feature{p}(1,1,i);
    end
    feature{p} = feature{p}';
end
clear p i k n temp_feature
gamma=zeros(a*b,3);
for iteration=2:50
    iteration
   
    dign=zeros(a*b,1);
    % E step. Evaluate the responsibilities using the current parameter values
    for n=1:a*b
        for k=1:3
            dign(n)=dign(n)+ pii{k}*normal(feature{n},mu{k},cov{k});
        end
        for k=1:3
            gamma(n,k)=pii{k}*normal(feature{n},mu{k},cov{k})/dign(n);
        end
    end
     sum(log(dign))
     log_lh(iteration)=sum(log(dign));
    % M step. Re-estimate the parameters using the current responsibilities
    N1= sum(gamma(:,1));
    N2= sum(gamma(:,2));
    N3= sum(gamma(:,3));
    pii{1}=N1/(a*b);
    pii{2}=N2/(a*b);
    pii{3}=N3/(a*b);
    temp1=[0 0 0]';
    temp2=[0 0 0]';
    temp3=[0 0 0]';
    for n=1:a*b
        temp1=temp1+gamma(n,1)*feature{n};
        temp2=temp2+gamma(n,2)*feature{n};
        temp3=temp3+gamma(n,3)*feature{n};
    end
    mu{1}=temp1/N1;
    mu{2}=temp2/N2;
    mu{3}=temp3/N3;
    clear temp1 temp2 temp3
    temp1=zeros(3,3);
    temp2=zeros(3,3);
    temp3=zeros(3,3);
    for n=1:a*b
        temp1=temp1+gamma(n,1)*(feature{n}-mu{1})*(feature{n}-mu{1})';
        temp2=temp2+gamma(n,2)*(feature{n}-mu{2})*(feature{n}-mu{2})';
        temp3=temp3+gamma(n,3)*(feature{n}-mu{3})*(feature{n}-mu{3})';
    end
    cov{1}=temp1/N1;
    cov{2}=temp2/N2;
    cov{3}=temp3/N3;

        
end
for i=1:a*b
[u v ] =max(gamma(i,:));
temp_im(i)=v;
end
out=reshape(temp_im,b,a);
imagesc(out')
figure,plot(log_lh(2:50))
mu{1}
mu{2}
mu{3}
pii
for i=1:3
    cov{i}
end


