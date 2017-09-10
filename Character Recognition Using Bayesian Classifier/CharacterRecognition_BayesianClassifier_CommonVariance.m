%Build a rudimentary pattern recognizer by making use of the Bayesian decision theory concepts discussed in class. To this goal, you are given training images of 3 characters in a folder named TrainCharacters.zip.
%There are 200 training images of size 128 x 128 for each character class. For evaluating the classifiers, you are provided 300 test images of size 128 x 128 in a separate folder TestCharacters.zip.
%Assume the samples to be generated from a multi dimensional Gaussian distribution, having class specific mean vectors mu_i. Consider each of the modelling schemes for computing
%the covariance matrix.
%The samples across all the character classes are pooled together to generate a common non diagonal covariance matrix Sigma.
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

%----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
%(ii)The samples across all the character classes are pooled together to generate a common non diagonal covariance matrix .
common_cov= (cov+ cov2+cov3)/3;
b1=0;
b2=0;
b3=0;
for i=1:100
    H1_e= (transpose(inv(common_cov + 0.1*eye(32*32,32*32))*mu))*image_column_vector_test_e{i}-0.5*(transpose(mu)*(inv(common_cov + 0.1*eye(32*32,32*32))))*mu;
    H2_e= (transpose(inv(common_cov + 0.1*eye(32*32,32*32))*mu2))*image_column_vector_test_e{i}-0.5*(transpose(mu2)*(inv(common_cov + 0.1*eye(32*32,32*32))))*mu2;
    H3_e= (transpose(inv(common_cov + 0.1*eye(32*32,32*32))*mu3))*image_column_vector_test_e{i}-0.5*(transpose(mu3)*(inv(common_cov + 0.1*eye(32*32,32*32))))*mu3;
    if ( ((H1_e-H2_e) >0 ) & ((H1_e-H3_e) > 0) )
        i
        disp('image belong to category e (actual category = e) ');
        b1=b1+1;
    end
     if ( ((H1_e-H2_e) <0 ) &((H2_e-H3_e)> 0) )
        i
        disp(' th image belong to category c (actual category = e)');
        b2=b2+1;
    end
     if ( ((H1_e-H3_e) <0 ) &((H2_e-H3_e) <0) )
        i
        disp(' th image belong to category i(actual category = e) ');
        b3=b3+1;
    end
end
b4=0;
b5=0;
b6=0;
for i=1:100
    H1_c= (transpose(inv(common_cov + 0.1*eye(32*32,32*32))*mu))*image_column_vector_test_c{i}-0.5*(transpose(mu)*(inv(common_cov + 0.1*eye(32*32,32*32))))*mu;
    H2_c= (transpose(inv(common_cov + 0.1*eye(32*32,32*32))*mu2))*image_column_vector_test_c{i}-0.5*(transpose(mu2)*(inv(common_cov + 0.1*eye(32*32,32*32))))*mu2;
    H3_c= (transpose(inv(common_cov + 0.1*eye(32*32,32*32))*mu3))*image_column_vector_test_c{i}-0.5*(transpose(mu3)*(inv(common_cov + 0.1*eye(32*32,32*32))))*mu3;
    if ( ((H1_c-H2_c) >0 ) & ((H1_c-H3_c) > 0) )
         i
         disp('image belong to category e (actual category = c) ');
         b4=b4+1;
    end
    
     if ( ((H1_c-H2_c) <0 ) &((H2_c-H3_c)> 0) )
        i
        disp(' th image belong to category c (actual category = c)');
        b5=b5+1;
    end
      if ( ((H1_c-H3_c) <0 ) &((H2_c-H3_c) <0) )
        i
        disp(' th image belong to category i(actual category = c) ');
        b6=b6+1;
    end
end

b7=0;
b8=0;
b9=0;
for i=1:100
    H1_eye= (transpose(inv(common_cov + 0.1*eye(32*32,32*32))*mu))*image_column_vector_test_eye{i}-0.5*(transpose(mu)*(inv(common_cov + 0.1*eye(32*32,32*32))))*mu;
    H2_eye= (transpose(inv(common_cov + 0.1*eye(32*32,32*32))*mu2))*image_column_vector_test_eye{i}-0.5*(transpose(mu2)*(inv(common_cov + 0.1*eye(32*32,32*32))))*mu2;
    H3_eye= (transpose(inv(common_cov + 0.1*eye(32*32,32*32))*mu3))*image_column_vector_test_eye{i}-0.5*(transpose(mu3)*(inv(common_cov + 0.1*eye(32*32,32*32))))*mu3;
    if ( ((H1_eye-H3_eye) <0 ) &((H2_eye-H3_eye) <0) )
        i
        disp(' th image belong to category i(actual category = i) ');
        b7=b7+1;
    end
    if ( ((H1_eye-H2_eye) >0 ) & ((H1_eye-H3_eye) > 0) )
        i
        disp('image belong to category e (actual category = i) ');
        
         b8=b8+1;
    end
    
    if ( ((H1_eye-H2_eye) <0 ) &((H2_eye-H3_eye)> 0) )
        i
        disp(' th image belong to category c (actual category = i)');
        b9=b9+1;
    end
    
end
