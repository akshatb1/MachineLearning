

%The gallery folder Gallery.zip contains images from 40 individuals, each of them providing 5 images. The pixel intensities of the 200 face images will be used for
%computing the KL Transform. By employing the method of efficient computation of the
%basis vectors for high dimensional data (discussed in class),
%(i) Display the Eigenface images corresponding to the top 5 Eigen values of the covariance
%matrix .
%(ii) Plot a graph depicting the percentage of the total variance of the original data
%retained in the reduced space versus the number of dimensions. From this graph,
%find the number of dimensions required for projecting the face vectors so that:
%(a) at least 85% of the total variance of the original data is accounted for in the
reduced space.
%(b) at least 95% of the total variance of the original data is accounted for in the
%reduced space.
%(iii) Reconstruct the image ‘face input 1.pgm’ using the:
%(a) Eigenface corresponding to the largest eigenvalue.
%(b) Top 4 Eigenfaces
%(c) Top 15 Eigenfaces
%(d) Top 150 Eigenfaces
%(e) All the Eigenfaces
%Display the reconstructed image and the mean squared error in each case.
%(iv) Depict graphically the mean squared error obtained for different number of Eigenfaces.
%Note that the Eigenfaces will vary from 1 to 200 (total number of training
%samples).
%(v) Repeat the parts (iii) and (iv) for the image ‘face input 2.pgm’. Comment on your
%result.



clc;
clear all   % Clear the command window.
myPath = 'Gallery\Gallery\gallery';
% Define a starting folder.
start_path = fullfile(myPath);
% Get list of all subfolders.
allSubFolders = genpath(myPath);
remain = allSubFolders;
listOfFolderNames = {};
while true
	[singleSubFolder, remain] = strtok(remain, ';');
	if isempty(singleSubFolder)
		break;
	end
	listOfFolderNames = [listOfFolderNames singleSubFolder];
end
numberOfFolders = length(listOfFolderNames);

%Process all image files in those folders and store them in a cell.
C = cell(200, 1);
for k = 2 : numberOfFolders
	thisFolder = listOfFolderNames{k};
	filePattern = sprintf('%s/*.pgm', thisFolder);
	baseFileNames = dir(filePattern);
    numberOfImageFiles = length(baseFileNames);
	if numberOfImageFiles >= 1
		% Go through all those image files.
		for f = 1 : numberOfImageFiles
			fullFileName = fullfile(thisFolder, baseFileNames(f).name);
            C{f+(5*(k-2))} = im2double(imread(fullFileName));
            
        end
    end
end
clear listOfFolderNames;
% Extracting Feature Vectors from Images : Creating a Feature Vector for
% Each Image
for i=1:200                                        
    featureVector{i} = (reshape((C{i})',1 ,112*92))';
    
end
clear C;
average= zeros(112*92, 1);                                   % Sum of all image column vector initialized as 0
for i=1:200
   average=average+ featureVector{i};
end
mu=average/200;
clear average;                                   

% Creating X matrix of Size N X D for high dimensional PCA Analysis
X= zeros(200,112*92);
for k=1:200
    X(k,:)= (featureVector{k}-mu)';
end
%Computing eigenvectors and eigenvalues of X*X'
[V, D] = eig((1/200)*(X*X'));
U=zeros(112*92,200);
for i=1:200
    U(:,i)=1/sqrt(200*D(i,i))*(X'*V(:,i));
end
% First Part : Displaying Top five eigenfaces :
for k=200:-1:196
    temp= sqrt(255)*U(:,k);
    temp2= temp+mu;
    temp3= (reshape(temp2',92,112)');
    figure,imshow(temp3)
end
%Second Part : Plotting captured variance v/s no. of dimensions used
a= (1/200)*trace((transpose(X))*X);
sum=0;
var=(1/200)*trace(X*X');
captured=zeros(1,200);
for k=200:-1:1
    sum=sum+D(k,k);
    captured(201-k)=sum/var;
end
figure, plot(captured);
grid on;
title('Fraction of Variance Captured v/s Reduced Dimensions');
ylabel('Fraction of Variance Captured');
xlabel('No. of Dimensions');
    
% % %----------------------------------------------------------------------------------------------------------------
%Part 3 and Part 4: Reconstruction and Mean Square Error Plot for Face 1
cd('');
face1=im2double(imread('face_input_1.pgm'));
feature1 = (reshape(face1',1 ,112*92))';
W=cell(5,1);
W{1}=U(:,200);                   %Tranformation Matrix for highest Eigenface
W{2}=U(:,200:-1:197);            %Transformation Matrix for top 4 Eigenfaces
W{3}=U(:,200:-1:186);            %Transformation Matrix for top 15 Eigenfaces
W{4}=U(:,200:-1:151);            %Transformation Matrix for top 150 Eigenfaces
W{5}=U(:,200:-1:1);              %Transformation Matrix for top 200 Eigenfaces
recons_image=cell(5,1);
mean_square_error=zeros(5,1);
for i=1:5
    z= (W{i})'*(feature1-mu);
    recons_feature1= (W{i})*z+mu;
    mean_square_error(i)=(norm(recons_feature1-feature1))^2;
    recons_image{i}= reshape(recons_feature1',92,112)';
    figure,imshow(recons_image{i});
    title('Reconstructed Image') ;
end
disp(' Normalized mean square errors for face 1 for various reconstructions:')
mean_square_error
Wnew=cell(200,1);
error_plot=zeros(1,200);
for i=200:-1:1
    Wnew{i}=U(:,200:-1:i);
    znew= (Wnew{i})'*(feature1-mu);
    recons_feature1new= (Wnew{i})*znew+mu;
    error_plot(201-i)=(norm(recons_feature1new-feature1))^2;
end
figure, plot( error_plot,'g');
grid on;
xlabel('No. of Eigenfaces');
ylabel('Mean Square Error');
title('Reconstruction of Face1: Normalized Mean Square Error v/s No. of Eigenface Plot');
clear Wnew
    
%-----------------------------------------------------------------------------------------------------------------------------
%Part 5 : Reconstruction and MSE Plot for Face 2
face2=im2double(imread('face_input_2.pgm'));
feature2 = (reshape(face2',1 ,112*92))';
W2=cell(5,1);
W2{1}=U(:,200);                   %Tranformation Matrix for highest Eigenface
W2{2}=U(:,200:-1:197);            %Transformation Matrix for top 4 Eigenfaces
W2{3}=U(:,200:-1:186);            %Transformation Matrix for top 15 Eigenfaces
W2{4}=U(:,200:-1:151);            %Transformation Matrix for top 150 Eigenfaces
W2{5}=U(:,200:-1:1);              %Transformation Matrix for top 200 Eigenfaces
recons_image2=cell(5,1);
mean_square_error2= zeros(5,1);
for i=1:5
    z2= (W2{i})'*(feature2-mu);
    recons_feature2= (W2{i})*z2+mu;
    mean_square_error2(i)=(norm(recons_feature2-feature2))^2;
    recons_image2{i}= reshape(recons_feature2',92,112)';
    figure,imshow(recons_image2{i});
    title('Reconstructed Image for Face2');
end
disp(' Normalized mean square errors for face 2 for various reconstructions:')
mean_square_error2
W2new=cell(200,1);
error_plot2=zeros(1,200);
for i=200:-1:1
    W2new{i}=U(:,200:-1:i);
    z2new= (W2new{i})'*(feature2-mu);
    recons_feature2new= (W2new{i})*z2new+mu;
    error_plot2(201-i)=(norm(recons_feature2new-feature2))^2;
end
figure,plot( error_plot2,'g');
grid on;
xlabel('No. of Eigenfaces');
ylabel('Mean Square Error');
title('Reconstruction of Face2:Normalized Mean Square Error v/s No. of Eigenface Plot');
clear W2new
%-------------------------------------------------------------------------------------------------------------------





