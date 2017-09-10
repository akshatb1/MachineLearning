%Multi class classification using Support Vector Machine and comparative analysis

clc
clear all
load('Pattern1.mat');
load('Pattern2.mat');
load('Pattern3.mat');
load('Test1.mat');
load('Test2.mat');
load('Test3.mat');
for i=1:200
    class1(i,:)= train_pattern_1{i};
    class2(i,:)=train_pattern_2{i};
    class3(i,:)=train_pattern_3{i};
end
Training= [class1;class2;class3];
%Class 1 v/s All
Group =[ones(200,1); 2*ones(400,1)];
it=1;
for c=0.1:0.5:40.1
    jt=1;
    for sigma=2:2:32
        SVMStruct = svmtrain(Training,Group,'boxconstraint',c,'Kernel_Function','rbf','rbf_sigma',sigma);
        for i=1:100
            test1(i,:)=test_pattern_1{i};
            test2(i,:)=test_pattern_2{i};
            test3(i,:)=test_pattern_3{i};
        end
        Sample=[test1;test2;test3];
        Group_out = svmclassify(SVMStruct,Sample);
        Accuracy1(it,jt)=0;
        for i=1:100
            if Group_out(i)==1
                Accuracy1(it,jt)=Accuracy1(it,jt)+1;
            end
        end
        %Class 2 v/s All
        Group_2 =[ones(200,1); 2*ones(200,1); ones(200,1)];
        SVMStruct = svmtrain(Training,Group_2,'boxconstraint',c,'Kernel_Function','rbf','rbf_sigma',sigma);
        Group_out_2 = svmclassify(SVMStruct,Sample);
        Accuracy2(it,jt)=0;
        for i=101:200
            if Group_out_2(i)==2
                Accuracy2(it,jt)=Accuracy2(it,jt)+1;
            end
        end
        %Class 3 v/s All
        %Class 2 v/s All
        Group_3 =[ones(200,1); ones(200,1); 3*ones(200,1)];
        SVMStruct = svmtrain(Training,Group_3,'boxconstraint',c,'Kernel_Function','rbf','rbf_sigma',sigma);
        Group_out_3 = svmclassify(SVMStruct,Sample);
        Accuracy3(it,jt)=0;
        for i=201:300
            if Group_out_3(i)==3
                Accuracy3(it,jt)=Accuracy3(it,jt)+1;
            end
        end
        jt=jt+1;
    end
    it=it+1;
end 
 
surf((Accuracy1+Accuracy2+Accuracy3)/3)
xlabel('Sigma Parameter of RBF (1 cm = 2units)(2:2:32)');
ylabel('Box Constraint of SVM(1 cm = 0.5)(0.1:0.5:40)');
zlabel('% Average Accuracy');
title('Variation of Average Accuracy')
 figure;surf(Accuracy1)
xlabel('Sigma Parameter of RBF (1 cm = 2units)(2:2:32)');
ylabel('Box Constraint of SVM(1 cm = 0.5)(0.1:0.5:40)');
zlabel('%  Accuracy1');
title('Variation of  Accuracy1')
figure; surf(Accuracy2)
xlabel('Sigma Parameter of RBF (1 cm = 2units)(2:2:32)');
ylabel('Box Constraint of SVM(1 cm = 0.5)(0.1:0.5:40)');
zlabel('%  Accuracy2');
title('Variation of  Accuracy2')
figure;surf(Accuracy3)
xlabel('Sigma Parameter of RBF (1 cm = 2units)(2:2:32)');
ylabel('Box Constraint of SVM(1 cm = 0.5)(0.1:0.5:40)');
zlabel('%  Accuracy3');
title('Variation of  Accuracy3')

