
%Build a rudimentary pattern recognizer by making use of the Bayesian decision theory concepts discussed in class. To this goal, you are given training images of 3 characters in a folder named TrainCharacters.zip.
%There are 200 training images of size 128 x 128 for each character class. For evaluating the classifiers, you are provided 300 test images of size 128 x 128 in a separate folder TestCharacters.zip.
%Assume the samples to be generated from a multi dimensional Gaussian distribution, having class specific mean vectors mu_i. Consider each of the modelling schemes for computing
%the covariance matrix.

%The covariance matrix of each class is forced to be spherical.


%Build a generative Bayesian classifier using the training images and categorize the 300 character samples contained in the test folder. The mean and the
%covariance matrices are to be estimated from the training data using the Maximum Likelihood techniques. Report the individual character accuracies as well as the averaged
%accuracy for each of the models.
%Employ the 128x128 pixel intensity values as features (after appropriate normalization). If you encounter memory storage issues during simulation, you may consider resizing the
%images to a more manageable size (say 32 x 32) for the feature computation. However, note that in order to beat the curse of dimensionality, you have to add a regularization
%term of the form lambda*I in the computation of the covariance matrix.



clc;
clear all;
%Read Input Training Images from data set
cd('D:\IITG Stuff\Acads\6th Sem\EE657 Pattern Recognition and Machine Learning\Assignments\ASSIGNMENT_1_PRML_2014\ASSIGNMENT_1_PRML_2014\TrainCharacters\TrainCharacters\1');
D = dir('*.jpg');
z=numel(D);
imcell = cell(1,300);                              % cell to read and store images from a folder
imcellnew=cell(1,300);                             % cell to store images after resizing and converting into double
for i = 1:200                                      % Loop to read images, resize them to 32*32 and convert into double data type
  imcell{i} = imread(D(i).name);
  I=imresize(imcell{i},0.25);                           % Resizing to 32*32
  J=im2double(I);                                       % Converting into Double
  imcellnew{i}=J;                                       % Storing in a new cell
end

clear  D I J;
cd('D:\IITG Stuff\Acads\6th Sem\EE657 Pattern Recognition and Machine Learning\Assignments\ASSIGNMENT_1_PRML_2014\ASSIGNMENT_1_PRML_2014\TestCharacters\TestCharacters\TestCharacters\1');
D_test_e = dir('*.jpg');
for i = 1:100                                      % Loop to read images, resize them to 32*32 and convert into double data type
  imcell{i+200} = imread(D_test_e(i).name);
  I=imresize(imcell{i},0.25);                           % Resizing to 32*32
  J=im2double(I);                                       % Converting into Double
  imcellnew{i+200}=J;                                       % Storing in a new cell
end
    
for i=1:z                                        % To create Column Vectors from Each Image Matrix
  image_column_vector{i} = reshape((imcellnew{i})',1 ,32*32)';
end
for i=1:100                                     % To create Column Vectors from Each Image Matrix
  image_column_vector_test_e{i} = reshape((imcellnew{i+200})',1 ,32*32)';
end
clear imcellnew;
clear imcell;
average= zeros(32*32, 1);                                   % Sum of all image column vector initialized as 0
for i=1:z
   average=average+ image_column_vector{i};
end

mu=average/200;
clear average;
                                       % Finding out mu using Maximum Likelihood = SUM(Xi)/No. of Samples
temp_cov = zeros(32*32,32*32);
for i=1:z
    temp_cov = temp_cov +(image_column_vector{i}-mu)*(transpose(image_column_vector{i}-mu));
end
cov = (1/200)*temp_cov;
clear temp_cov image_column_vector;
%------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------;
cd('D:\IITG Stuff\Acads\6th Sem\EE657 Pattern Recognition and Machine Learning\Assignments\ASSIGNMENT_1_PRML_2014\ASSIGNMENT_1_PRML_2014\TrainCharacters\TrainCharacters\2');
D2 = dir('*.jpg');
imcell2 = cell(1,z);                              % cell to read and store images from a folder
imcellnew2=cell(1,z);                             % cell to store images after resizing and converting into double
for i = 1:z                                      % Loop to read images, resize them to 32*32 and convert into double data type
  imcell2{i} = imread(D2(i).name);
  I2=imresize(imcell2{i},0.25);                           % Resizing to 32*32
  J2=im2double(I2);                                       % Converting into Double
  imcellnew2{i}=J2;                                       % Storing in a new cell
end
clear D2 imcell2;
clear I2,J2;
for i=1:z                                        % To create Column Vectors from Each Image Matrix
  image_column_vector2{i} = reshape((imcellnew2{i})',1 ,32*32)';
end
clear imcellnew2;

average2= zeros(32*32, 1);                                % Sum of all image column vector initialized as 0
for i=1:z
   average2=average2+ image_column_vector2{i};
end

mu2=average2/200;    % Finding out mu using Maximum Likelihood = SUM(Xi)/No. of Samples

clear average2;
temp_cov2 = zeros(32*32,32*32);
for i=1:z
    temp_cov2 = temp_cov2 +(image_column_vector2{i}-mu2)*(transpose(image_column_vector2{i}-mu2));
end
cov2 = (1/200)*temp_cov2;
clear temp_cov2 image_column_vector2;
%----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------;
%Read Input Training Images from data set
cd('D:\IITG Stuff\Acads\6th Sem\EE657 Pattern Recognition and Machine Learning\Assignments\ASSIGNMENT_1_PRML_2014\ASSIGNMENT_1_PRML_2014\TrainCharacters\TrainCharacters\3');
D3 = dir('*.jpg');
imcell3 = cell(1,z);                              % cell to read and store images from a folder
imcellnew3=cell(1,z);                             % cell to store images after resizing and converting into double
for i = 1:z                                      % Loop to read images, resize them to 32*32 and convert into double data type
  imcell3{i} = imread(D3(i).name);
  I3=imresize(imcell3{i},0.25);                           % Resizing to 32*32
  J3=im2double(I3);                                       % Converting into Double
  imcellnew3{i}=J3;                                       % Storing in a new cell
end
for i=1:z                                        % To create Column Vectors from Each Image Matrix
  image_column_vector3{i} = reshape((imcellnew3{i})',1 ,32*32)';
end
clear imcell3;
clear I3,J3;
clear imcellnew3 D3;
average3= zeros(32*32, 1);                                   % Sum of all image column vector initialized as 0
for i=1:z
   average3=average3+ image_column_vector3{i};
end

mu3=average3/200;                                             % Finding out mu using Maximum Likelihood = SUM(Xi)/No. of Samples
clear average3;

temp_cov3 = zeros(32*32,32*32);
for i=1:z
    temp_cov3 = temp_cov3 +(image_column_vector3{i}-mu3)*(transpose(image_column_vector3{i}-mu3));
end
cov3 = (1/200)*temp_cov3;
clear temp_cov3 image_column_vector3;



%------------------------------------------------------------------------------------------------------------------------------------------------------------
%Loading 'c'test images
cd('D:\IITG Stuff\Acads\6th Sem\EE657 Pattern Recognition and Machine Learning\Assignments\ASSIGNMENT_1_PRML_2014\ASSIGNMENT_1_PRML_2014\TestCharacters\TestCharacters\TestCharacters\2');
D_test_c = dir('*.jpg');
x=100;
imcell_test_c    = cell(1,x);                              % cell to read and store images from a folder
imcellnew_test_c = cell(1,x);
for i = 1:x                                      % Loop to read images, resize them to 32*32 and convert into double data type
  imcell_test_c{i} = imread(D_test_c(i).name);
  I_test_c=imresize(imcell_test_c{i},0.25);              % Resizing to 32*32
  J_test_c=im2double(I_test_c);                            % Converting into Double
  imcellnew_test_c{i}=J_test_c;                                       % Storing in a new cell
end
clear D_test_c;
for i=1:x                                        % To create Column Vectors from Each Image Matrix
  image_column_vector_test_c{i} = reshape((imcellnew_test_c{i})',1 ,32*32)';
end
clear imcell_test_c imcellnew_test_c;

%Loading 'i' test images
cd('D:\IITG Stuff\Acads\6th Sem\EE657 Pattern Recognition and Machine Learning\Assignments\ASSIGNMENT_1_PRML_2014\ASSIGNMENT_1_PRML_2014\TestCharacters\TestCharacters\TestCharacters\3');
D_test_eye = dir('*.jpg');
imcell_test_eye    = cell(1,x);                              % cell to read and store images from a folder
imcellnew_test_eye = cell(1,x);
for i = 1:x                                      % Loop to read images, resize them to 32*32 and convert into double data type
  imcell_test_eye{i} = imread(D_test_eye(i).name);
  I_test_eye=imresize(imcell_test_eye{i},0.25);              % Resizing to 32*32
  J_test_eye=im2double(I_test_eye);                            % Converting into Double
  imcellnew_test_eye{i}=J_test_eye;                                       % Storing in a new cell
end
clear D_test_eye ;
for i=1:x                                       % To create Column Vectors from Each Image Matrix
  image_column_vector_test_eye{i} = reshape((imcellnew_test_eye{i})',1 ,32*32)';
end
clear imcell_test_eye, imcellnew_test_eye;
%------------------------------------------------------------------------------------------------------------------------------------------------------
% (iv)The samples across all the character classes are pooled to generate a common diagonal covariance matrix  The diagonal entries correspond to the variances of the
% individual features, that are considered to be independent.
coviii=cov;
for i=1:1024
    for j=1:1024
        if i~= j;       
            coviii(i,j)=0;
        end
    end
end
coviii2=cov2;
for i=1:1024
    for j=1:1024
        if i~= j;       
            coviii2(i,j)=0;
        end
    end
end
coviii3=cov3;
for i=1:1024
    for j=1:1024
        if i~= j;       
            coviii3(i,j)=0;
        end
    end
end
cov_fourth=(coviii+ coviii2 +coviii3) /3;

coviii;
coviii2;
coviii3;
temp_sigma=0;
temp_sigma2=0;
temp_sigma3=0;
for i=1:1024
    temp_sigma=temp_sigma+coviii(i,i);
end
cov_v = (1/1024)*temp_sigma*eye(1024,1024);

for i=1:1024
    temp_sigma2=temp_sigma2+coviii2(i,i);
end
cov_v2=(1/1024)*temp_sigma2*eye(1024,1024);

for i=1:1024
    temp_sigma3=temp_sigma3+coviii3(i,i);
end
cov_v3=(1/1024)*temp_sigma3*eye(1024,1024);
%Classifying 'e' characters
l_count1=0;
l_count2=0;
l_count3=0;
for i=1:100
    L1 =(transpose(image_column_vector_test_e{i}))*(-0.5*(inv(cov_v + eye(32*32,32*32))))*(image_column_vector_test_e{i}) + (transpose((inv(cov_v+eye(32*32,32*32)))*mu))*image_column_vector_test_e{i} - 0.5*(transpose(mu)*((inv(cov_v + eye(32*32,32*32))))*mu) -0.5*(log(det(cov_v + eye(32*32,32*32))));
    L2 =(transpose(image_column_vector_test_e{i}))*(-0.5*(inv(cov_v2+ eye(32*32,32*32))))*(image_column_vector_test_e{i}) + (transpose((inv(cov_v2+eye(32*32,32*32)))*mu2))*image_column_vector_test_e{i} - 0.5*(transpose(mu2)*((inv(cov_v2 + eye(32*32,32*32))))*mu2) -0.5*(log(det(cov_v2 + eye(32*32,32*32))));
    L3 =(transpose(image_column_vector_test_e{i}))*(-0.5*(inv(cov_v3 + eye(32*32,32*32))))*(image_column_vector_test_e{i}) + (transpose((inv(cov_v3+eye(32*32,32*32)))*mu3))*image_column_vector_test_e{i} - 0.5*(transpose(mu3)*((inv(cov_v3 + eye(32*32,32*32))))*mu3) -0.5*(log(det(cov_v3 + eye(32*32,32*32))));
    if ( ((L1-L2) >0 ) & ((L1-L3) > 0) )
        i
        disp('image belong to category e (actual category = e) ');
        l_count1=l_count1+1;
    end
     if ( ((L1-L2) <0 ) &((L2-L3)> 0) )
        i
        disp(' th image belong to category c (actual category = e)');
        l_count2=l_count2+1;
    end
     if ( ((L1-L3) <0 ) &((L2-L3) <0) )
        i
        disp(' th image belong to category i(actual category = e) ');
        l_count3=l_count3+1;
    end
end
%Classifying 'c' characters
l_count4=0;
l_count5=0;
l_count6=0;
for i=1:100
    L1_c =(transpose(image_column_vector_test_c{i}))*(-0.5*(inv(cov_v + eye(32*32,32*32))))*(image_column_vector_test_c{i}) + (transpose((inv(cov_v+eye(32*32,32*32)))*mu))*image_column_vector_test_c{i} - 0.5*(transpose(mu)*((inv(cov_v + eye(32*32,32*32))))*mu) -0.5*(log(det(cov_v + eye(32*32,32*32))));
    L2_c =(transpose(image_column_vector_test_c{i}))*(-0.5*(inv(cov_v2 + eye(32*32,32*32))))*(image_column_vector_test_c{i}) + (transpose((inv(cov_v2+eye(32*32,32*32)))*mu2))*image_column_vector_test_c{i} - 0.5*(transpose(mu2)*((inv(cov_v2 + eye(32*32,32*32))))*mu2) -0.5*(log(det(cov_v2 + eye(32*32,32*32))));
    L3_c =(transpose(image_column_vector_test_c{i}))*(-0.5*(inv(cov_v3 + eye(32*32,32*32))))*(image_column_vector_test_c{i}) + (transpose((inv(cov_v3+eye(32*32,32*32)))*mu3))*image_column_vector_test_c{i} - 0.5*(transpose(mu3)*((inv(cov_v3 + eye(32*32,32*32))))*mu3) -0.5*(log(det(cov_v3 + eye(32*32,32*32))));
    if ( ((L1_c-L2_c) >0 ) & ((L1_c-L3_c) > 0) )
        i
        disp('image belong to category e (actual category = c) ');
       
        l_count4=l_count4+1;
    end
     if ( ((L1_c-L2_c) <0 ) &((L2_c-L3_c)> 0) )
        i
        disp(' th image belong to category c (actual category = c)');
        
        l_count5=l_count5+1;
    end
    if ( ((L1_c-L3_c) <0 ) &((L2_c-L3_c) <0) )
        i
        disp(' th image belong to category i(actual category = c) ');
        l_count6=l_count6+1;
    end
end


%classifying 'i' character
l_count7=0;
l_count8=0;
l_count9=0;
for i=1:100
    L1_1 =(transpose(image_column_vector_test_eye{i}))*(-0.5*(inv(cov_v +  eye(32*32,32*32))))*(image_column_vector_test_eye{i}) + (transpose((inv(cov_v+eye(32*32,32*32)))*mu))*image_column_vector_test_eye{i} - 0.5*(transpose(mu)*((inv(cov_v + eye(32*32,32*32))))*mu) -0.5*(log(det(cov_v + eye(32*32,32*32))));
    L2_1 =(transpose(image_column_vector_test_eye{i}))*(-0.5*(inv(cov_v2 + eye(32*32,32*32))))*(image_column_vector_test_eye{i}) + (transpose((inv(cov_v2+eye(32*32,32*32)))*mu2))*image_column_vector_test_eye{i} - 0.5*(transpose(mu2)*((inv(cov_v2 + eye(32*32,32*32))))*mu2) -0.5*(log(det(cov_v2 + eye(32*32,32*32))));
    L3_1 =(transpose(image_column_vector_test_eye{i}))*(-0.5*(inv(cov_v3 + eye(32*32,32*32))))*(image_column_vector_test_eye{i}) + (transpose((inv(cov_v3+eye(32*32,32*32)))*mu3))*image_column_vector_test_eye{i} - 0.5*(transpose(mu3)*((inv(cov_v3 + eye(32*32,32*32))))*mu3) -0.5*(log(det(cov_v3 + eye(32*32,32*32))));
    if ( ((L1_1-L2_1) >0 ) & ((L1_1-L3_1) > 0) )
        i
        disp('image belong to category e (actual category = i) ');
        l_count7=l_count7+1;
    end
    if ( ((L1_1-L2_1) <0 ) &((L2_1-L3_1)> 0) )
        i
        disp(' th image belong to category c (actual category = i)');
         l_count8=l_count8+1;
    end
    if ( ((L3_1 - L1_1) >0 ) &((L3_1-L2_1) >0) )
        i
        disp(' th image belong to category i(actual category = i) ');
         l_count9=l_count9+1;
    end
end