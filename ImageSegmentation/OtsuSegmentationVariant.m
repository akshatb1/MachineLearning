%Written by Akshat Bordia (akshat.bordia31@gmail.com)
%A variant of an image segmentation/ clustering technique, popularly known in the literature as : Otsu segmentation. The idea of this
%algorithm is to automatically select a threshold pixel intensity for separating the pixel intensities of the image to one of 2 categories : foreground (set to 255) and background (set
%to 0). The choice of the threshold is made based on the Fisher score criterion, but here
%you would work directly on the pixel domain. Essentially, you are required to compute
%the Fisher scores for each of the gray level intensities in the image, and then consider the
%one having the maximum value as the threshold.
%Test the algorithm on the image cameraman.jpg. Display the segmented image and the threshold used

clc
clear all
%Automatic Thresholding Based on Fishers Score%
%Please keep the code and image in same folder before running code.
filename = uigetfile('*.jpg','Please provide the camerman image to perfrom the segmentation :');
image=imread(filename);
[Counts, X]= imhist(image);
cov1=zeros(256,1);
cov2=zeros(256,1);
for i=0:255
    mu1(i+1) =(sum(X(1:i+1,1).*Counts(1:i+1,1)))/(sum(Counts(1:i+1,1)));
    mu2(i+1) =(sum(X(i+2 : 256,1).*Counts(i+2:256,1)))/(sum(Counts(i+2 : 256,1)));
    cov1(i+1)=  sum((((X(1:i+1,1)).^2 ).*(Counts(1:i+1,1))))/(sum(Counts(1:i+1,1))) -(mu1(i+1))^2;
    cov2(i+1)=  sum((((X(i+1:256,1)).^2 ).*(Counts(i+1:256,1))))/(sum(Counts(i+2 : 256,1)))-(mu2(i+1))^2;
    FG(i+1) = (mu1(i+1) -mu2(i+1))^2/(cov1(i+1)+cov2(i+1));      %Computing Fisher Score for Each Value of threshold
    WC(i+1)= (sum(Counts(1:i+1,1)))*cov1(i+1) +(sum(Counts(i+2 : 256,1)))*cov2(i+1);
end
[a,b]= max(FG);
disp('The threshold determined using Fisher Score Criterion is :')
threshold = b-1
[c,d] = min(WC);
disp('The threshold determined using Weighted Within Class Variance(Ostu Algorithm) is :')

threshold_weighted = d-1

for i=1:512
    for j=1:512
        if image(i,j)<=b-1
            image(i,j)=0;
        else
            image(i,j)=255;
        end
    end
end
imshow(image)
title ('Segmented Image using Automatic Thresholding Technique')