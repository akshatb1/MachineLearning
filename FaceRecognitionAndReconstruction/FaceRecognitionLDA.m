%Written by Akshat Bordia (akshat.bordia31@gmail.com)
%implement a Fisher Linear Discriminant for classifying the face
%images from the probe folder. Employ the principal components obtained using the 200
%Eigenfaces from Assignment 2 as the features. Project the data on to
%10 significant eigenvectors
% 25 significant eigenvectors
% 39 significant eigenvectors
%Report the accuracy using the 3-nearest neighbor classier for each case.

clc;
clear all   % Clear the command window.
myPath = uigetdir('','Please provide the ..\Gallery\Gallery folder (folder where subfolders s1,s2.. are placed)');
start_path = fullfile(myPath);
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
%---------------------------------------------------------------------------------
%Process all image files in those folders.
C1 = cell(200, 1);
for k = 2 : numberOfFolders
	% Get this folder and print it out.
	thisFolder = listOfFolderNames{k};
	
	% Get PNG files.
	filePattern = sprintf('%s/*.pgm', thisFolder);
	baseFileNames = dir(filePattern);
    numberOfImageFiles = length(baseFileNames);
	if numberOfImageFiles >= 1
		% Go through all those image files.
		for f = 1 : numberOfImageFiles
			fullFileName = fullfile(thisFolder, baseFileNames(f).name);
			%fprintf('     Processing image file %s\n', fullFileName);
            C1{f+(5*(k-2))} = im2double(imread(fullFileName));
            
        end
    end
end
clear listOfFolderNames;
% Extracting Feature Vectors from Images : Creating a Feature Vector for
% Each Image


    
for i=1:200                                        
    featureVector{i} = (reshape((C1{i})',1 ,112*92))';
    
end
clear C1
average= zeros(112*92, 1);                                   % Sum of all image column vector initialized as 0
for i=1:200
   average=average+ featureVector{i};
end
mu=average/200;
clear average
%Creating Matrix for PCA
for k=1:200
    X(k,:)= (featureVector{k}-mu)';
end
[V, D] = eig((1/200)*(X*X'));
U=zeros(112*92,200);
for i=1:200
    U(:,i)=1/sqrt(200*D(i,i))*(X'*V(:,i));          %Final Eigenvector Matrix in original space 
end
%Finding projection values of each image onto top 200 eigenfaces
W=U;
Z=cell(200,1);
for i=1:200
    Z{i}= W'*(featureVector{i}-mu);             %Finding Projection value onto 200 most significant directions
end
    
%---------------------------------------------------------------------------
%Loading Test Data and Testing it
myPath2 = uigetdir('','Please Provide Path for Test Image Folder : \Assignment\Probe\Probe folder');
% Define a starting folder.
start_path2 = fullfile(myPath2);
% Get list of all subfolders.
allSubFolders2 = genpath(myPath2);
% Parse into a cell array.
remain2 = allSubFolders2;
listOfFolderNames2 = {};
while true
	[singleSubFolder2, remain2] = strtok(remain2, ';');
	if isempty(singleSubFolder2)
		break;
	end
	listOfFolderNames2 = [listOfFolderNames2 singleSubFolder2];
end
numberOfFolders2 = length(listOfFolderNames2);

%Process all image files in those folders.
C2 = cell(200, 1);
for k = 2 : numberOfFolders2
	thisFolder2 = listOfFolderNames2{k};
	filePattern2 = sprintf('%s/*.pgm', thisFolder2);
	baseFileNames2 = dir(filePattern2);
    numberOfImageFiles2 = length(baseFileNames2);
	if numberOfImageFiles2 >= 1
		% Go through all those image files.
		for f = 1 : numberOfImageFiles2
			fullFileName2 = fullfile(thisFolder2, baseFileNames2(f).name);
			%fprintf('     Processing image file %s\n', fullFileName);
            C2{f+(5*(k-2))} = im2double(imread(fullFileName2));
        end
    end
end
clear listOfFolderNames2;
% Extracting Feature Vectors from Images : Creating a Feature Vector for
% Each Image
for i=1:200                                        
    featureVector2{i} = (reshape((C2{i})',1 ,112*92))';
end
Ztest=cell(200,1);
for i=1:200
    Ztest{i}= W'*(featureVector2{i}-mu);             %Finding Projection value onto 25 most significant directions
end
%%Upto this part we have done PCA Analysis on the data to reduce it to 200
%%dimensions. This was same as it was done in Assignment 2. Now we have to
%%apply Fisher Linear Discriminant on the reduced data.
mean =zeros(200,40);
for i=1:40
    mean(:,i)= (Z{1+(i-1)*5}+Z{2+(i-1)*5}+Z{3++(i-1)*5}+Z{4++(i-1)*5}+Z{5++(i-1)*5})/5;
    cov{i} = (Z{1+(i-1)*5}-mean(:,i))*(Z{1+(i-1)*5}-mean(:,i))' +(Z{2+(i-1)*5}-mean(:,i))*(Z{2+(i-1)*5}-mean(:,i))'+(Z{3+(i-1)*5}-mean(:,i))*(Z{3+(i-1)*5}-mean(:,i))' +(Z{4+(i-1)*5}-mean(:,i))*(Z{4+(i-1)*5}-mean(:,i))'+(Z{5+(i-1)*5}-mean(:,i))*(Z{5+(i-1)*5}-mean(:,i))';
end
total_mean = sum(mean,2)/40;
within_scatter =zeros(200,200);
bw_scatter = zeros(200,200);
for i=1:40
within_scatter = within_scatter + cov{i};
bw_scatter =bw_scatter + 5*(mean(:,i)-total_mean)*(mean(:,i)-total_mean)';
end
%Fisher Linear Discriminant: Finding Fisher Components
[W,E] = eig((inv(within_scatter+0.01*eye(200))*bw_scatter));

% Task2 : First Part : 10 significant eigenvectors
P1=W(:,1:10);
for i=1:200
   Y{i}= P1'*(Z{i});
   training(i,:) = Y{i}';
end
for i=1:200
   Ytest{i}= P1'*(Ztest{i});
   test(i,:)= Ytest{i}';
end
result =knnsearch(real(training),real(test),'k',3);

for i =1:200
    for j=1:3
        class(i,j)= 1+ floor((result(i,j)-1)/5);
    end
end
count =0;
for i=1:200
    if (class(i,1) ==class(i,2)== 1+ floor((i-1)/5))||(class(i,2) ==class(i,3)== 1+ floor((i-1)/5))||(class(i,1) ==class(i,3)== 1+ floor((i-1)/5))
        count =count+1;
    elseif class(i,1)==1+floor((i-1)/5)
        count=count+1;
    end
end
disp(' % Accuracy using 10 significant eigenvectors is :')
accuracy_10 =count/2

% For 25 Significant Eigenvectors
P1=W(:,1:25);
for i=1:200
   Y{i}= P1'*(Z{i});
   training2(i,:) = Y{i}';
end
for i=1:200
   Ytest{i}= P1'*(Ztest{i});
   test2(i,:)= Ytest{i}';
end
result =knnsearch(real(training2),real(test2),'k',3);

for i =1:200
    for j=1:3
        class(i,j)= 1+ floor((result(i,j)-1)/5);
    end
end
count =0;
for i=1:200
    if (class(i,1) ==class(i,2)== 1+ floor((i-1)/5))||(class(i,2) ==class(i,3)== 1+ floor((i-1)/5))||(class(i,1) ==class(i,3)== 1+ floor((i-1)/5))
        count =count+1;
    elseif class(i,1)==1+floor((i-1)/5)
        count=count+1;
    end
end
disp(' % Accuracy using 25 significant eigenvectors is :')
accuracy_25 =count/2

% For 39 Eigenvectors
P1=W(:,1:39);
for i=1:200
   Y{i}= P1'*(Z{i});
   training3(i,:) = Y{i}';
end
for i=1:200
   Ytest{i}= P1'*(Ztest{i});
   test3(i,:)= Ytest{i}';
end
result =knnsearch(real(training3),real(test3),'k',3);

for i =1:200
    for j=1:3
        class(i,j)= 1+ floor((result(i,j)-1)/5);
    end
end
count =0;
for i=1:200
    if (class(i,1) ==class(i,2)== 1+ floor((i-1)/5))||(class(i,2) ==class(i,3)== 1+ floor((i-1)/5))||(class(i,1) ==class(i,3)== 1+ floor((i-1)/5))
        count =count+1;
    elseif class(i,1)==1+floor((i-1)/5)
        count=count+1;
    end
end
disp(' % Accuracy using 39 significant eigenvectors is :')
accuracy_39 =count/2

