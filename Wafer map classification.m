%https://www.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html
%% Data Load
clc
clear all
close all
imds = imageDatastore('E:\programing\Research]\simple CNN Code', ...
    'IncludeSubfolders', true, 'Labelsource', 'foldernames');

%% Displaying some of the images in the datastore
figure;
perm = randperm(2951,20); %total number of MRI 2951 & I will show here 20 MRI
for i = 1:20
  subplot(4,5,i);
   imshow(imds.Files{perm(i)});
end
%Calculating the number of images in each category
labelCount = imds.countEachLabel;

%specifying the size of the images in the input layer of the network
img = readimage(imds,1);
size(img)


%% Dividing the data into training and validation data sets
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,'randomize'); %randomly 80% data will be sellected for traning & test of the data will go for validation

imageSize = [32 32 1]; %input image size


%% Defining the convolutional neural network architecture
layers = [
    imageInputLayer([32 32 1]) % input layer declaration 

    convolution2dLayer(3, 36,'Padding','same')  %1st Conv layer %4*4 kernel & 32 number of filter
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2) 

    convolution2dLayer(4, 86,'Padding','same') %2nd Conv layer
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2) 

    convolution2dLayer(3,146,'Padding','same') %3rd Conv layer
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2) 

   convolution2dLayer(3,186,'Padding','same') %4rd Conv layer
   batchNormalizationLayer
   reluLayer
% additionLayer(2,'Name','add')

   
%    convolution2dLayer(3,248,'Padding','same') %3rd Conv layer
 %   batchNormalizationLayer
  %  reluLayer

   % maxPooling2dLayer(2,'Stride',2) 
    
    % convolution2dLayer(4,496,'Padding','same') %3rd Conv layer
   % batchNormalizationLayer
   % reluLayer

   % maxPooling2dLayer(2,'Stride',2) 
    
      

%try

%convolution2dLayer(3, 124,'Padding','same') %4th Conv layer
%batchNormalizationLayer
%reluLayer
    fullyConnectedLayer(100)

    fullyConnectedLayer(100)
   
    fullyConnectedLayer(3)%output layer for 3 types of Tumors
    softmaxLayer  %output activation function
    classificationLayer];


%% Specifying the training options
options = trainingOptions('sgdm', ...
'InitialLearnRate',0.01, ...
'MaxEpochs',5, ...
'Shuffle','every-epoch', ...
'ValidationData',imdsValidation, ...
'ValidationFrequency',10, ...
'Verbose',false, ...
'Plots','training-progress');

%Training the network using the architecture defined by layers
net = trainNetwork(imdsTrain,layers,options);


%% analyzeNetwork(net)
% calculate the final validation accuracy
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
%confusion matrix creating
plotconfusion(YPred,YValidation); 
accuracy = sum(YPred == YValidation)/numel(YValidation)%calculation of accuracy

