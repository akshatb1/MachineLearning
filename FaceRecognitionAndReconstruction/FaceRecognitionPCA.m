
%The test image folder Probe.zip contains 5 images of each of the 40
%individuals.
%(a) Classify the test samples in this folder by a 1-nearest neighbor classifier (with
%Euclidean distance) in a reduced 25 dimensional subspace. Compute the classification
%accuracy.
%(b) Depict graphically the recognition accuracies obtained for different number of
%dimensions. For this part, you have to vary the dimensions from 1 to 200 (total
%number of training samples).


clc;
clear all   % Clear the command window.
myPath = '\Gallery\Gallery\gallery';
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
%Finding projection values of each image onto top 25 eigenfaces
W=U(:,200:-1:176);
Z=cell(200,1);
for i=1:200
    Z{i}= W'*(featureVector{i}-mu);             %Finding Projection value onto 25 most significant directions
end
    
%---------------------------------------------------------------------------
%Loading Test Data and Testing it
myPath2 = '\Probe\Probe';
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
%Classifying each image in new projections
distance=zeros(200,200);
for i=1:200
    for k=1:200
        distance(k,i)= norm((Ztest{i}-Z{k}));
    end
end
result=zeros(200,1);
for i=1:200             % Finding Minimum Distance and assigning it to that category 
  [a,b] = min(distance(:,i)); 
  result(i)=b;          % Result i is the number of training image, the test image is assigned to, It must be 
end
correct_count=0;
for k=0:39
    for i=(1+5*k) :(5+5*k)
        if(1+5*k <= (result(i)) &&(result(i))<= 5+5*k)
            i
            disp( 'th image is classified correctly')
            correct_count=correct_count+1;
        else
            i
            disp('th image is NOT classified correctly')
        end
    end
end
Accuracy = (correct_count/200)*100
%---------------------------------------------------------------------------
% %Plot of recognition accuracy v/s dimensions
dist_cell=cell(200,1);
res_cell=cell(200,1);
count=zeros(200,1);
for i=200:-1:1
     TM=U(:,200:-1:i);
     for k=1:200
         Znew{k}= TM'*(featureVector{k}-mu); 
         Ztest_new{k}= TM'*(featureVector2{k}-mu);
     end
     for j=1:200
         for l=1:200
             dist_cell{i}(j,l)= norm((Ztest_new{j}-Znew{l}));
         end
     end
     for m=1:200            
         [a,b] = min(dist_cell{i}(:,m)); 
          res_cell{i}(m)=b;          
     end
     for n=0:39
         for p=(1+5*n) :(5+5*n)
            if(1+5*n <= (res_cell{i}(p)) &&(res_cell{i}(p))<= 5+5*n)
            count(201-i)=count(201-i)+1;
            end
    	
         end
     end
end
count=count/2;
figure,plot(count,'g');
title('Face Recognition Accuracy v/s no. of dimension');
xlabel('No. of dimensions');
ylabel(' % Accuracy of Recognition');

