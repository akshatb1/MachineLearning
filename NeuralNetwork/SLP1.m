%Single Layer Perceptron
%1. By using the features contained in Pattern1.mat and Pattern2.mat, design a
%“batch-mode” perceptron classifier for the classes !1 and !2. How many iterations
%are required for convergence. Evaluate the performance on the test data Test1.mat
%and Test2.mat and report the accuracy.
%2. Now consider building a perceptron for the classes !1 and !3. How many iterations
%are required for convergence in this case. Comment on the result. Evaluate the
%performance of this classifier on Test1.mat and Test3.mat.




clc
clear all
load('Pattern1.mat');
load('Pattern2.mat');
for i=1:200
    feature1{i}=[ 1 train_pattern_1{i}]';
    feature1{i+200}=-1*[1 (train_pattern_2{i})]'; % Normalization
end
%initialize a
a=(1/2)*ones(121,1);
for p=1:1500
    count=1;
    temp=zeros(121,1);
    sum(p)=0;
    for i=1:400
        sum(p)=sum(p)+a'*feature1{i};
    end
    for i=1:400
        if a'*feature1{i} < 0
            temp = temp+feature1{i};
            count=count+1;
        end
    end
    a=a+0.01*temp;
    
end
plot(-sum);
hold on , title('Plot depicting convergence of Objective Function for Perceptron')
xlabel('No. of Iterations')
ylabel('Perceptron Criterion Function')
load('Test1.mat');
load('Test2.mat');
for i=1:100
    feature2{i}=[ 1 test_pattern_1{i}]';
    feature2{i+100}=-1*[1 (test_pattern_2{i})]';
end
accuracy1=0;
for i=1:100
    if a'*feature2{i}>= 0
        accuracy1=accuracy1+1;
    end
end
disp('Accuracy for class-1 for classification b/w class 1 and 2 is :')
accuracy1
accuracy2=0;
for i=1:100
    if a'*feature2{i+100}>=0
        accuracy2=accuracy2+1;
    end
end
disp('Accuracy for class-2 for classification b/w class 1 and 2 is :')
accuracy2
