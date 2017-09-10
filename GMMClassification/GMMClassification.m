%Gaussian Mixture Model as a density estimation technique to the problem of classiffication.
%You have been provided training features, corresponding to two classes !1 and !2
%in the files Pattern1.mat and Pattern2.mat respectively. Each le contains 200
%instances (training examples), of 120 feature dimensions.
%The features corresponding to 100 testing samples of !1 and !2 are contained in Test1.mat and Test2.mat respectively.
%1. Utilize the features contained in Pattern1.mat and Pattern2.mat to build separate
%Gaussian Mixture Models for each class !1 and !2. You may choose 2 Gaussian
%components for each mixture. The initial means may be randomly chosen from the
%training data by running the k-means algorithm. The covariance matrix for each
%Gaussian component can be initialized to the identity matrix.
%2. Use the trained models from part (1) to test the performance of the features con-
%tained in the files Test1.mat and Test2.mat. Report the recognition accuracy for each class.

clc
clear all
load('Pattern1.mat');
%Extracting Data from .mat file
for n=1:200
    training(n,:)= train_pattern_1{n};
end
% Step 1 : Initialization of means using K-means Algorithm and
% initialization of Covariance and Pi_k (Weights)
[idx, mu]=kmeans(training,2);
cov{1}= 0.1*eye(120);
cov{2}= 0.1*eye(120);
mu_cell{1}=mu(1,:)';
mu_cell{2}=mu(2,:)';
gamma=zeros(200,2);
pii{1}=0.5;
pii{2}=0.5;
%Estimating GMM Parameters using EM Algorithms
for iteration=1:20
    iteration
    %Step 2 : E Step
    resp = zeros(200,1);
    for n=1:200
        
        for k=1:2
            resp(n)= resp(n)+ pii{k}*normal(training(n,:)',mu_cell{k},cov{k});
        end
        clear k
        for k =1:2
            gamma(n,k)= pii{k}*normal(training(n,:)',mu_cell{k},cov{k})/resp(n);
        end
    end
    clear mu_cell cov 
    %Step 3: M step
    
        N1= sum(gamma(:,1));
        N2= sum(gamma(:,2));
        pii{1}=N1/200;
        pii{2}=N2/200;
        pii;
        temp1=zeros(120,1);
        temp2=zeros(120,1);
        for n=1:200
            temp1= temp1+gamma(n,1)*training(n,:)';
            temp2= temp2+gamma(n,2)*training(n,:)';
        end
        mu_cell{1}=temp1/N1;
        mu_cell{2}= temp2/N2;
        clear temp1 temp2
        temp1=zeros(120);
        temp2=zeros(120);
        for n=1:200
            temp1=temp1+ gamma(n,1)*(training(n,:)'-mu_cell{1})*(training(n,:)'-mu_cell{1})';
            temp2=temp2+ gamma(n,2)*(training(n,:)'-mu_cell{2})*(training(n,:)'-mu_cell{2})';
        end
        cov{1} = temp1/N1 + 0.005*eye(120); 
        cov{2} = temp2/N2 + 0.005*eye(120); 
        
        
        
        log_lh(iteration)= sum(log(resp));
end
%--------------------------------------------------------------------------
%Loading and Doing EM Estimation for class 2%
load('Pattern2.mat');
for n=1:200
    training2(n,:)= train_pattern_2{n};
end
% Step 1 : Initialization
[idx, mu2]=kmeans(training2,2);
cov2{1}= 0.1*eye(120);
cov2{2}= 0.1*eye(120);
mu_cell2{1}=mu2(1,:)';
mu_cell2{2}=mu2(2,:)';
gamma2=zeros(200,2);
pii2{1}=0.5;
pii2{2}=0.5;

for iteration=1:20
    iteration
    %Step 2 : E Step
    resp2 = zeros(200,1);
    for n=1:200
        
        for k=1:2
            resp2(n)= resp2(n)+ pii2{k}*normal(training2(n,:)',mu_cell2{k},cov2{k});
        end
        clear k
        for k =1:2
            gamma2(n,k)= pii2{k}*normal(training2(n,:)',mu_cell2{k},cov2{k})/resp2(n);
        end
    end
    clear mu_cell2 cov2 
    %Step 3: M step
    
        N12= sum(gamma2(:,1));
        N22= sum(gamma2(:,2));
        pii2{1}=N12/200;
        pii2{2}=N22/200;
        pii2;
        temp12=zeros(120,1);
        temp22=zeros(120,1);
        for n=1:200
            temp12= temp12+gamma2(n,1)*training2(n,:)';
            temp22= temp22+gamma2(n,2)*training2(n,:)';
        end
        mu_cell2{1}=temp12/N12;
        mu_cell2{2}= temp22/N22;
        clear temp12 temp22
        temp12=zeros(120);
        temp22=zeros(120);
        for n=1:200
            temp12=temp12+ gamma2(n,1)*(training2(n,:)'-mu_cell2{1})*(training2(n,:)'-mu_cell2{1})';
            temp22=temp22+ gamma2(n,2)*(training2(n,:)'-mu_cell2{2})*(training2(n,:)'-mu_cell2{2})';
        end
        cov2{1} = temp12/N12 + 0.005*eye(120); 
        cov2{2} = temp22/N22 + 0.005*eye(120); 
        
        log_lh2(iteration)= sum(log(resp2));
end
%--------------------------------------------------------------------------
% % End of Training Part %
% % Testing Part
load('Test1.mat')
for n=1:100
    test_1(n,:)= test_pattern_1{n};
end
%Now we need to calculate two class conditional density function values.
G1=zeros(100,1);
G2=zeros(100,1);
count=0;
%Using Baye's Classifier for classification using 2-Component GMM
for n=1:100
    for k=1:2
        G1(n)= G1(n)+ pii{k}*normal(test_1(n,:)',mu_cell{k},cov{k});
        G2(n)= G2(n)+ pii2{k}*normal(test_1(n,:)',mu_cell2{k},cov2{k});
        
    end
    if G1(n) > G2(n)
        count=count+1;
    end
        
end
disp('Accuracy for first class is :')
accuracy1=count

% % %Testing w2 class
load('Test2.mat');
for n=1:100
    test_2(n,:) =test_pattern_2{n};
end
H1=zeros(100,1);
H2=zeros(100,1);
count2=0;
for n=1:100
    for k=1:2
        H1(n) =H1(n)+ pii{k}*normal(test_2(n,:)',mu_cell{k},cov{k});
        H2(n) =H2(n)+ pii2{k}*normal(test_2(n,:)',mu_cell2{k},cov2{k});
    end
    if H2(n) >H1(n)
        count2=count2+1;
    end
end
disp('Accuracy for second class is :')
accuracy2= count2
disp('Average Accuracy is :')
accuracy= (accuracy1+accuracy2)/2