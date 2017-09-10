
%Build a rudimentary pattern recognizer by making use of the Bayesian decision theory concepts discussed in class. To this goal, you are given training images of 3 characters in a folder named TrainCharacters.zip.
%There are 200 training images of size 128 x 128 for each character class. For evaluating the classifiers, you are provided 300 test images of size 128 x 128 in a separate folder TestCharacters.zip.
%Assume the samples to be generated from a multi dimensional Gaussian distribution, having class specific mean vectors mu_i. Consider each of the modelling schemes for computing
%the covariance matrix.



%Build a generative Bayesian classifier using the training images and categorize the 300 character samples contained in the test folder. The mean and the
%covariance matrices are to be estimated from the training data using the Maximum Likelihood techniques. Report the individual character accuracies as well as the averaged
%accuracy for each of the models.

% Features : In this method, for each training image, we calculate following quantities for Black and White Image (converted):
%1. For every row, horizontal count of transition from 0 to 1;
%2. For every column, vertical count of transition from 0 to1 ;
%3. For every row, the nearest pixel which is high value(1) (from right and left);
%4. For every column, the nearest pixel which is high value(1)(from top and bottom);




clc;
clear all;
%Read Input Training Images from data set
cd('\TrainCharacters\TrainCharacters\1');
D = dir('*.jpg');
z=numel(D);
imcell = cell(1,z);                              % cell to read and store images from a folder
imcellnew=cell(1,z);                             % cell to store images after resizing and converting into double
for k = 1:z                                      % Loop to read images, resize them to 32*32 and convert into double data type
  imcell{k} = imread(D(k).name);
  P=im2bw(imcell{k});
  I=imresize(P,0.25);                           % Resizing to 32*32
  J=im2double(I);                                       % Converting into Double
  imcellnew{k}=J;                                       % Storing in a new cell
end
clear imcell D;
clear  i k P I J;
verti_crossings      = cell(1,z);
hori_crossings       = cell(1,z);
verti_distance_up    = cell(1,z);
verti_distance_down  = cell(1,z);
hori_distance_back   = cell(1,z);
hori_distance_front  = cell(1,z);
feature_vector       = cell(1,z);
%detecting crossings of an image
for k =1:z 
    temp_image =imcellnew{k};
    [m,n]=size(temp_image);
    verti_crossings{k}      =   zeros(1,32);
    hori_crossings {k}      =   zeros(1,32);
    verti_distance_up{k}    =   zeros(1,32);
    verti_distance_down{k}  =   zeros(1,32);
    hori_distance_back{k}   =   zeros(1,32);
    hori_distance_front{k}  =   zeros(1,32);
    for i=1:m
        for j=2:n
            if ((temp_image(i,j)-temp_image(i,j-1)) >0)||(-(temp_image(i,j)-temp_image(i,j-1)) > 0)
                hori_crossings{k}(j)=hori_crossings{k}(j)+1;
            end
        end
    end
    for j=1:n
        for i=2:m
            if ((temp_image(i,j)-temp_image(i-1,j)) >0)||(-(temp_image(i,j)-temp_image(i-1,j)) > 0)
                verti_crossings{k}(i)= verti_crossings{k}(i)+1;
            end
        
        end
    end
    for i=1:32
        for j=1:32
            if temp_image(i,j)>0
                hori_distance_front{k}(i)=j; %#ok<*SAGROW>
                break;
            end
        end
    end
    for i=1:32
        for j=32:-1:1
            if temp_image(i,j)>0
                hori_distance_back{k}(i)=j; %#ok<*SAGROW>
                break;
            end
        end
    end
    for j=1:32
        for i=1:32
            if temp_image(i,j)>0
                verti_distance_up{k}(j)=i; %#ok<*SAGROW>
                break;
            end
        end
    end
    for j=1:32
        for i=32:-1:1
            if temp_image(i,j)>0
                verti_distance_down{k}(j)=i; %#ok<*SAGROW>
                break;
            end
        end
    end
    feature_vector{k} =[(hori_crossings{k})';(verti_crossings{k})';(hori_distance_front{k})';(hori_distance_back{k})';(verti_distance_up{k})';(verti_distance_down{k})'];

end

clear imcellnew temp_image hori_crossings verti_crossings hori_distance_front hori_distance_back verti_distance_up verti_distance_down;
average=zeros(192,1);
for k=1:z
   average=average+ feature_vector{k};
end

mu=average/200;
clear average;
                                       % Finding out mu using Maximum Likelihood = SUM(Xi)/No. of Samples
temp_cov = zeros(192,192);
for i=k:z
    temp_cov = temp_cov +(feature_vector{k}-mu)*(transpose(feature_vector{k}-mu));
end
cov = (1/200)*temp_cov;
clear temp_cov;
clear feature_vector m n ;

%Read Input Training Images from data set : c characters

%-------------------------------------------------------------------------------------------------------------------------------------------------------
cd('\TrainCharacters\TrainCharacters\2');
D2 = dir('*.jpg');
imcell2 = cell(1,z);                              % cell to read and store images from a folder
imcellnew2=cell(1,z);                             % cell to store images after resizing and converting into double
for k = 1:z                                      % Loop to read images, resize them to 32*32 and convert into double data type
  imcell2{k} = imread(D2(k).name);
  P2=im2bw(imcell2{k});
  I2=imresize(P2,0.25);                           % Resizing to 32*32
  J2=im2double(I2);                                       % Converting into Double
  imcellnew2{k}=J2;                                       % Storing in a new cell
end
clear imcell2;
clear  D2 P2 I2;
clear J2;
verti_crossings2      = cell(1,z);
hori_crossings2       = cell(1,z);
verti_distance_up2    = cell(1,z);
verti_distance_down2  = cell(1,z);
hori_distance_back2   = cell(1,z);
hori_distance_front2  = cell(1,z);
feature_vector2       = cell(1,z);
%detecting crossings of an image
for k =1:z 
    temp_image2 =imcellnew2{k};
    [m,n]=size(temp_image2);
    verti_crossings2{k}      =   zeros(1,32);
    hori_crossings2{k}       =   zeros(1,32);
    verti_distance_up2{k}    =   zeros(1,32);
    verti_distance_down2{k}  =   zeros(1,32);
    hori_distance_back2{k}   =   zeros(1,32);
    hori_distance_front2{k}  =   zeros(1,32);
    for i=1:m
        for j=2:n
            if ((temp_image2(i,j)-temp_image2(i,j-1)) >0)||(-(temp_image2(i,j)-temp_image2(i,j-1)) > 0)
                hori_crossings2{k}(j)=hori_crossings2{k}(j) + 1;
            end
        end
    end
    for j=1:n
        for i=2:m
            if ((temp_image2(i,j)-temp_image2(i-1,j)) >0)||(-(temp_image2(i,j)-temp_image2(i-1,j)) > 0)
                verti_crossings2{k}(i)= verti_crossings2{k}(i)+1;
            end
        
        end
    end
    for i=1:32
        for j=1:32
            if temp_image2(i,j)>0
                hori_distance_front2{k}(i)=j; %#ok<*SAGROW>
                break;
            end
        end
    end
    for i=1:32
        for j=32:-1:1
            if temp_image2(i,j)>0
                hori_distance_back2{k}(i)=j; %#ok<*SAGROW>
                break;
            end
        end
    end
    for j=1:32
        for i=1:32
            if temp_image2(i,j)>0
                verti_distance_up2{k}(j)=i; %#ok<*SAGROW>
                break;
            end
        end
    end
    for j=1:32
        for i=32:-1:1
            if temp_image2(i,j)>0
                verti_distance_down2{k}(j)=i; %#ok<*SAGROW>
                break;
            end
        end
    end
    feature_vector2{k} =[(hori_crossings2{k})';(verti_crossings2{k})';(hori_distance_front2{k})';(hori_distance_back2{k})';(verti_distance_up2{k})';(verti_distance_down2{k})'];

end

clear imcellnew;
average2=zeros(192,1);
for k=1:z
   average2=average2+ feature_vector2{k};
end

mu2=average2/200;
clear average2;
                                       % Finding out mu using Maximum Likelihood = SUM(Xi)/No. of Samples
temp_cov2 = zeros(192,192);
for i=k:z
    temp_cov2 = temp_cov2 +(feature_vector2{k}-mu2)*(transpose(feature_vector2{k}-mu2));
end
cov2 = (1/200)*temp_cov2;
clear temp_cov2;
clear feature_vector2;  
clear m n imcellnew2 temp_image2 hori_crossings2 verti_crossings2 hori_distance_front2 hori_distance_back2 verti_distance_up2 verti_distance_down2;
%Reading Input Images from Folder3 (i)
cd('\TrainCharacters\TrainCharacters\3');
D3 = dir('*.jpg');
imcell3 = cell(1,z);                              % cell to read and store images from a folder
imcellnew3=cell(1,z);                             % cell to store images after resizing and converting into double
for k = 1:z                                      % Loop to read images, resize them to 32*32 and convert into double data type
  imcell3{k} = imread(D3(k).name);
  P3=im2bw(imcell3{k});
  I3=imresize(P3,0.25);                           % Resizing to 32*32
  J3=im2double(I3);                                       % Converting into Double
  imcellnew3{k}=J3;                                       % Storing in a new cell
end
clear D3 imcell3;
clear  i  P3 I3 J3;

verti_crossings3      = cell(1,z);
hori_crossings3       = cell(1,z);
verti_distance_up3    = cell(1,z);
verti_distance_down3  = cell(1,z);
hori_distance_back3   = cell(1,z);
hori_distance_front3  = cell(1,z);
feature_vector3       = cell(1,z);
%detecting crossings of an image
for k =1:z 
    temp_image3 =imcellnew3{k};
    [m,n]=size(temp_image3);
    verti_crossings3{k}      =   zeros(1,32);
    hori_crossings3 {k}      =   zeros(1,32);
    verti_distance_up3{k}    =   zeros(1,32);
    verti_distance_down3{k}  =   zeros(1,32);
    hori_distance_back3{k}   =   zeros(1,32);
    hori_distance_front3{k}  =   zeros(1,32);
    for i=1:m
        for j=2:n
            if ((temp_image3(i,j)-temp_image3(i,j-1)) >0)||(-(temp_image3(i,j)-temp_image3(i,j-1)) > 0)
                hori_crossings3{k}(j)=hori_crossings3{k}(j)+1;
            end
        end
    end
    for j=1:n
        for i=2:m
            if ((temp_image3(i,j)-temp_image3(i-1,j)) >0)||(-(temp_image3(i,j)-temp_image3(i-1,j)) > 0)
                verti_crossings3{k}(i)= verti_crossings3{k}(i)+1;
            end
        
        end
    end
    for i=1:32
        for j=1:32
            if temp_image3(i,j)>0
                hori_distance_front3{k}(i)=j; %#ok<*SAGROW>
                break;
            end
        end
    end
    for i=1:32
        for j=32:-1:1
            if temp_image3(i,j)>0
                hori_distance_back3{k}(i)=j; %#ok<*SAGROW>
                break;
            end
        end
    end
    for j=1:32
        for i=1:32
            if temp_image3(i,j)>0
                verti_distance_up3{k}(j)=i; %#ok<*SAGROW>
                break;
            end
        end
    end
    for j=1:32
        for i=32:-1:1
            if temp_image3(i,j)>0
                verti_distance_down3{k}(j)=i; %#ok<*SAGROW>
                break;
            end
        end
    end
    feature_vector3{k} =[(hori_crossings3{k})';(verti_crossings3{k})';(hori_distance_front3{k})';(hori_distance_back3{k})';(verti_distance_up3{k})';(verti_distance_down3{k})'];

end

clear imcellnew;
average3=zeros(192,1);
for k=1:z
   average3=average3+ feature_vector3{k};
end

mu3=average3/200;
clear average3;
                                       % Finding out mu using Maximum Likelihood = SUM(Xi)/No. of Samples
temp_cov3 = zeros(192,192);
for i=k:z
    temp_cov3 = temp_cov3 +(feature_vector3{k}-mu3)*(transpose(feature_vector3{k}-mu3));
end
cov3 = (1/200)*temp_cov3;
clear temp_cov3;
clear feature_vector3 m n;  
clear imcellnew3 temp_image3 hori_crossings3 verti_crossings3 hori_distance_front3 hori_distance_back3 verti_distance_up3 verti_distance_down3;
%------------------------------------------------------------------------------------------------------------------------------------------------------------
%Reading Test Data to extract features from them in order to classify images
cd('\TestCharacters\TestCharacters\TestCharacters\1');
D_test_e = dir('*.jpg');
x= numel(D_test_e);
imcell_test_e = cell(1,x);                              % cell to read and store images from a folder
imcellnew_test_e=cell(1,x);                             % cell to store images after resizing and converting into double
for k = 1:x                                      % Loop to read images, resize them to 32*32 and convert into double data type
  imcell_test_e{k} = imread(D_test_e(k).name);
  P_test_e=im2bw(imcell_test_e{k});
  I_test_e=imresize(P_test_e,0.25);                           % Resizing to 32*32
  J_test_e=im2double(I_test_e);                                       % Converting into Double
  imcellnew_test_e{k}=J_test_e;                                       % Storing in a new cell
end
clear D_test_e imcell_test_e;
clear P_test_e I_test_e J_test_e;
verti_crossings_test_e      = cell(1,x);
hori_crossings_test_e       = cell(1,x);
verti_distance_up_test_e    = cell(1,x);
verti_distance_down_test_e  = cell(1,x);
hori_distance_back_test_e   = cell(1,x);
hori_distance_front_test_e  = cell(1,x);
feature_vector_test_e       = cell(1,x);
%detecting crossings of an image
for k =1:x 
    temp_image_test_e =imcellnew_test_e{k};
    [m,n]=size(temp_image_test_e);
    verti_crossings_test_e{k}      =   zeros(1,32);
    hori_crossings_test_e {k}      =   zeros(1,32);
    verti_distance_up_test_e{k}    =   zeros(1,32);
    verti_distance_down_test_e{k}  =   zeros(1,32);
    hori_distance_back_test_e{k}   =   zeros(1,32);
    hori_distance_front_test_e{k}  =   zeros(1,32);
    for i=1:m
        for j=2:n
            if ((temp_image_test_e(i,j)-temp_image_test_e(i,j-1)) >0)||(-(temp_image_test_e(i,j)-temp_image_test_e(i,j-1)) > 0)
                hori_crossings_test_e{k}(j)=hori_crossings_test_e{k}(j)+1;
            end
        end
    end
    for j=1:n
        for i=2:m
            if ((temp_image_test_e(i,j)-temp_image_test_e(i-1,j)) >0)||(-(temp_image_test_e(i,j)-temp_image_test_e(i-1,j)) > 0)
                verti_crossings_test_e{k}(i)= verti_crossings_test_e{k}(i)+1;
            end
        
        end
    end
    for i=1:32
        for j=1:32
            if temp_image_test_e(i,j)>0
                hori_distance_front_test_e{k}(i)=j; %#ok<*SAGROW>
                break;
            end
        end
    end
    for i=1:32
        for j=32:-1:1
            if temp_image_test_e(i,j)>0
                hori_distance_back_test_e{k}(i)=j; %#ok<*SAGROW>
                break;
            end
        end
    end
    for j=1:32
        for i=1:32
            if temp_image_test_e(i,j)>0
                verti_distance_up_test_e{k}(j)=i; %#ok<*SAGROW>
                break;
            end
        end
    end
    for j=1:32
        for i=32:-1:1
            if temp_image_test_e(i,j)>0
                verti_distance_down_test_e{k}(j)=i; %#ok<*SAGROW>
                break;
            end
        end
    end
    feature_vector_test_e{k} =[(hori_crossings_test_e{k})';(verti_crossings_test_e{k})';(hori_distance_front_test_e{k})';(hori_distance_back_test_e{k})';(verti_distance_up_test_e{k})';(verti_distance_down_test_e{k})'];

end
clear imcellnew_test_e temp_image_test_e hori_crossings_test_e verti_crossings_test_e  hori_distance_front_test_e hori_distance_back_test_e verti_distance_up_test_e verti_distance_down_test_e;
%Reading Test Data Set of Images with character 'C'
cd('\TestCharacters\TestCharacters\TestCharacters\3');
D_test_i = dir('*.jpg');
x;
imcell_test_i = cell(1,x);                              % cell to read and store images from a folder
imcellnew_test_i=cell(1,x);                             % cell to store images after resizing and converting into double
for k = 1:x                                      % Loop to read images, resize them to 32*32 and convert into double data type
  imcell_test_i{k} = imread(D_test_i(k).name);
  P_test_i=im2bw(imcell_test_i{k});
  I_test_i=imresize(P_test_i,0.25);                           % Resizing to 32*32
  J_test_i=im2double(I_test_i);                                       % Converting into Double
  imcellnew_test_i{k}=J_test_i;                                       % Storing in a new cell
end
clear D_test_i imcell_test_i;
clear P_test_i I_test_i J_test_i;
verti_crossings_test_i      = cell(1,x);
hori_crossings_test_i      = cell(1,x);
verti_distance_up_test_i    = cell(1,x);
verti_distance_down_test_i  = cell(1,x);
hori_distance_back_test_i   = cell(1,x);
hori_distance_front_test_i  = cell(1,x);
feature_vector_test_i       = cell(1,x);
%detecting crossings of an image
for k =1:x 
    temp_image_test_i =imcellnew_test_i{k};
    [m,n]=size(temp_image_test_i);
    verti_crossings_test_i{k}      =   zeros(1,32);
    hori_crossings_test_i{k}      =   zeros(1,32);
    verti_distance_up_test_i{k}    =   zeros(1,32);
    verti_distance_down_test_i{k}  =   zeros(1,32);
    hori_distance_back_test_i{k}   =   zeros(1,32);
    hori_distance_front_test_i{k}  =   zeros(1,32);
    for i=1:m
        for j=2:n
            if ((temp_image_test_i(i,j)-temp_image_test_i(i,j-1)) >0)||(-(temp_image_test_i(i,j)-temp_image_test_i(i,j-1)) > 0)
                hori_crossings_test_i{k}(j)=hori_crossings_test_i{k}(j)+1;
            end
        end
    end
    for j=1:n
        for i=2:m
            if ((temp_image_test_i(i,j)-temp_image_test_i(i-1,j)) >0)||(-(temp_image_test_i(i,j)-temp_image_test_i(i-1,j)) > 0)
                verti_crossings_test_i{k}(i)= verti_crossings_test_i{k}(i)+1;
            end
        
        end
    end
    for i=1:32
        for j=1:32
            if temp_image_test_i(i,j)>0
                hori_distance_front_test_i{k}(i)=j; %#ok<*SAGROW>
                break;
            end
        end
    end
    for i=1:32
        for j=32:-1:1
            if temp_image_test_i(i,j)>0
                hori_distance_back_test_i{k}(i)=j; %#ok<*SAGROW>
                break;
            end
        end
    end
    for j=1:32
        for i=1:32
            if temp_image_test_i(i,j)>0
                verti_distance_up_test_i{k}(j)=i; %#ok<*SAGROW>
                break;
            end
        end
    end
    for j=1:32
        for i=32:-1:1
            if temp_image_test_i(i,j)>0
                verti_distance_down_test_i{k}(j)=i; %#ok<*SAGROW>
                break;
            end
        end
    end
    feature_vector_test_i{k} =[(hori_crossings_test_i{k})';(verti_crossings_test_i{k})';(hori_distance_front_test_i{k})';(hori_distance_back_test_i{k})';(verti_distance_up_test_i{k})';(verti_distance_down_test_i{k})'];

end
clear imcellnew_test_c temp_image_test_c hori_crossings_test_c verti_crossings_test_c  hori_distance_front_test_c hori_distance_back_test_c verti_distance_up_test_c verti_distance_down_test_c;
%Reading Test Data Set of Images with character 'I'
cd('\TestCharacters\TestCharacters\TestCharacters\2');
D_test_c = dir('*.jpg');
x;
imcell_test_c = cell(1,x);                              % cell to read and store images from a folder
imcellnew_test_c=cell(1,x);                             % cell to store images after resizing and converting into double
for k = 1:x                                      % Loop to read images, resize them to 32*32 and convert into double data type
  imcell_test_c{k} = imread(D_test_c(k).name);
  P_test_c=im2bw(imcell_test_c{k});
  I_test_c=imresize(P_test_c,0.25);                           % Resizing to 32*32
  J_test_c=im2double(I_test_c);                                       % Converting into Double
  imcellnew_test_c{k}=J_test_c;                                       % Storing in a new cell
end
clear D_test_c imcell_test_c;
clear P_test_c I_test_c J_test_c;
verti_crossings_test_c      = cell(1,x);
hori_crossings_test_c       = cell(1,x);
verti_distance_up_test_c    = cell(1,x);
verti_distance_down_test_c  = cell(1,x);
hori_distance_back_test_c   = cell(1,x);
hori_distance_front_test_c  = cell(1,x);
feature_vector_test_c       = cell(1,x);
%detecting crossings of an image
for k =1:x 
    temp_image_test_c =imcellnew_test_c{k};
    [m,n]=size(temp_image_test_c);
    verti_crossings_test_c{k}      =   zeros(1,32);
    hori_crossings_test_c{k}      =   zeros(1,32);
    verti_distance_up_test_c{k}    =   zeros(1,32);
    verti_distance_down_test_c{k}  =   zeros(1,32);
    hori_distance_back_test_c{k}   =   zeros(1,32);
    hori_distance_front_test_c{k}  =   zeros(1,32);
    for i=1:m
        for j=2:n
            if ((temp_image_test_c(i,j)-temp_image_test_c(i,j-1)) >0)||(-(temp_image_test_c(i,j)-temp_image_test_c(i,j-1)) > 0)
                hori_crossings_test_c{k}(j)=hori_crossings_test_c{k}(j)+1;
            end
        end
    end
    for j=1:n
        for i=2:m
            if ((temp_image_test_c(i,j)-temp_image_test_c(i-1,j)) >0)||(-(temp_image_test_c(i,j)-temp_image_test_c(i-1,j)) > 0)
                verti_crossings_test_c{k}(i)= verti_crossings_test_c{k}(i)+1;
            end
        
        end
    end
    for i=1:32
        for j=1:32
            if temp_image_test_c(i,j)>0
                hori_distance_front_test_c{k}(i)=j; %#ok<*SAGROW>
                break;
            end
        end
    end
    for i=1:32
        for j=32:-1:1
            if temp_image_test_c(i,j)>0
                hori_distance_back_test_c{k}(i)=j; %#ok<*SAGROW>
                break;
            end
        end
    end
    for j=1:32
        for i=1:32
            if temp_image_test_c(i,j)>0
                verti_distance_up_test_c{k}(j)=i; %#ok<*SAGROW>
                break;
            end
        end
    end
    for j=1:32
        for i=32:-1:1
            if temp_image_test_c(i,j)>0
                verti_distance_down_test_c{k}(j)=i; %#ok<*SAGROW>
                break;
            end
        end
    end
    feature_vector_test_c{k} =[(hori_crossings_test_c{k})';(verti_crossings_test_c{k})';(hori_distance_front_test_c{k})';(hori_distance_back_test_c{k})';(verti_distance_up_test_c{k})';(verti_distance_down_test_c{k})'];

end
clear imcellnew_test_c temp_image_test_c hori_crossings_test_c verti_crossings_test_c  hori_distance_front_test_c hori_distance_back_test_c verti_distance_up_test_c verti_distance_down_test_c;
%--------------------------------------------------------------------------------------------------------------------------------------------------------------
%Classifying 'e' characters
q_count1=0;
q_count2=0;
q_count3=0;
for i=1:x
    G1 =(transpose(feature_vector_test_e{i}))*(-0.5*(inv(cov + 0.3*eye(192,192))))*(feature_vector_test_e{i}) + (transpose((inv(cov+0.3*eye(192,192)))*mu))*feature_vector_test_e{i} - 0.5*(transpose(mu)*((inv(cov + 0.3*eye(192,192))))*mu) -0.5*(log(det(cov + 0.3*eye(192,192))));
    G2 =(transpose(feature_vector_test_e{i}))*(-0.5*(inv(cov2 + 0.3*eye(192,192))))*(feature_vector_test_e{i}) + (transpose((inv(cov2+0.3*eye(192,192)))*mu2))*feature_vector_test_e{i} - 0.5*(transpose(mu2)*((inv(cov2 + 0.3*eye(192,192))))*mu2) -0.5*(log(det(cov2 + 0.3*eye(192,192))));
    G3 =(transpose(feature_vector_test_e{i}))*(-0.5*(inv(cov3 + 0.3*eye(192,192))))*(feature_vector_test_e{i}) + (transpose((inv(cov3+0.3*eye(192,192)))*mu3))*feature_vector_test_e{i} - 0.5*(transpose(mu3)*((inv(cov3 + 0.3*eye(192,192))))*mu3) -0.5*(log(det(cov3 + 0.3*eye(192,192))));
    if ( ((G1-G2) >0 ) && ((G1-G3) > 0) )
        i
        disp('image belong to category e (actual category = e) ');
        q_count1= q_count1+1;
    end
     if ( ((G1-G2) <0 ) &((G2-G3)> 0) )
        i
        disp(' th image belong to category c (actual category = e)');
        q_count2= q_count2+1;
    end
     if ( ((G1-G3) <0 ) &((G2-G3) <0) )
        i
        disp(' th image belong to category i(actual category = e) ');
        q_count3= q_count3+1;
    end
end
%Clssifying "C" character
q_count4=0;
q_count5=0;
q_count6=0;
for i=1:x
    G1_c =(transpose(feature_vector_test_c{i}))*(-0.5*(inv(cov + 0.1*eye(192,192))))*(feature_vector_test_c{i}) + (transpose((inv(cov+0.1*eye(192,192)))*mu))*feature_vector_test_c{i} - 0.5*(transpose(mu)*((inv(cov + 0.1*eye(192,192))))*mu) -0.5*(log(det(cov + 0.1*eye(192,192))));
    G2_c =(transpose(feature_vector_test_c{i}))*(-0.5*(inv(cov2 + 0.1*eye(192,192))))*(feature_vector_test_c{i}) + (transpose((inv(cov2+0.1*eye(192,192)))*mu2))*feature_vector_test_c{i} - 0.5*(transpose(mu2)*((inv(cov2 + 0.1*eye(192,192))))*mu2) -0.5*(log(det(cov2 + 0.1*eye(192,192))));
    G3_c =(transpose(feature_vector_test_c{i}))*(-0.5*(inv(cov3 + 0.1*eye(192,192))))*(feature_vector_test_c{i}) + (transpose((inv(cov3+0.1*eye(192,192)))*mu3))*feature_vector_test_c{i} - 0.5*(transpose(mu3)*((inv(cov3 + 0.1*eye(192,192))))*mu3) -0.5*(log(det(cov3 + 0.1*eye(192,192))));
    if ( ((G1_c-G2_c) >0 ) && ((G1_c-G3_c) > 0) )
        i
        disp('image belong to category e (actual category = c) ');
        q_count4=q_count4+1;
       
    end
     if ( ((G1_c-G2_c) <0 ) &&((G2_c-G3_c)> 0) )
        i
        disp(' th image belong to category c (actual category = c)');
        q_count5=q_count5+1;
    end
    if ( ((G1_c-G3_c) <0 ) &&((G2_c-G3_c) <0) )
        i
        disp(' th image belong to category i(actual category = c) ');
        q_count6=q_count6+1;
        
    end
end
q_count7=0;
q_count8=0;
q_count9=0;
for i=1:x
    G1_i =(transpose(feature_vector_test_i{i}))*(-0.5*(inv(cov + 0.1*eye(192,192))))*(feature_vector_test_i{i}) + (transpose((inv(cov+0.1*eye(192,192)))*mu))*feature_vector_test_i{i} - 0.5*(transpose(mu)*((inv(cov + 0.1*eye(192,192))))*mu) -0.5*(log(det(cov + 0.1*eye(192,192))));
    G2_i =(transpose(feature_vector_test_i{i}))*(-0.5*(inv(cov2 + 0.1*eye(192,192))))*(feature_vector_test_i{i}) + (transpose((inv(cov2+0.1*eye(192,192)))*mu2))*feature_vector_test_i{i} - 0.5*(transpose(mu2)*((inv(cov2 + 0.1*eye(192,192))))*mu2) -0.5*(log(det(cov2 + 0.1*eye(192,192))));
    G3_i =(transpose(feature_vector_test_i{i}))*(-0.5*(inv(cov3 + 0.1*eye(192,192))))*(feature_vector_test_i{i}) + (transpose((inv(cov3+0.1*eye(192,192)))*mu3))*feature_vector_test_i{i} - 0.5*(transpose(mu3)*((inv(cov3 + 0.1*eye(192,192))))*mu3) -0.5*(log(det(cov3 + 0.1*eye(192,192))));
    if ( ((G1_i-G2_i) >0 ) && ((G1_i-G3_i) > 0) )
        i
        disp('image belong to category e (actual category = i) ');
        q_count7=q_count7+1;
       
    end
     if ( ((G1_i-G2_i) <0 ) &&((G2_i-G3_i)> 0) )
        i
        disp(' th image belong to category c (actual category = i)');
        q_count8=q_count8+1;
    end
    if ( ((G1_i-G3_i) <0 ) &&((G2_i-G3_i) <0) )
        i %#ok<*NOPTS>
        disp(' th image belong to category i(actual category = i) ');
        q_count9=q_count9+1;
        
    end
end
clear imcellnew_test_i temp_image_test_i hori_crossings_test_i verti_crossings_test_i  hori_distance_front_test_i hori_distance_back_test_i verti_distance_up_test_i verti_distance_down_test_i;