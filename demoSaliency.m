%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% This is a demo to compute the iseel saliency model
%%%
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc
clear
close all

% set the input and output folders and the extention of input files
stimuli = './Data/stimuli/';
resultFolder =  './Data/output/';
ext = 'jpg'; 

% add the needed toolboxes to the path
run('./utils/matconvnet/matlab/vl_setupnn');
addpath(genpath('./utils/elm'));
addpath(genpath('./utils/gistdescriptor'));

% load the network
net = load('deepModel/imagenet-vgg-verydeep-16.mat'); 
params.feat = 'VGG16'; 


% load the ensemble and set the number of learners you want
load('ensemble_net_osie');
nLearners = 490;


% load the stimuli images
fileList = dir([stimuli '*.' ext]);
params.fileList = fileList;

% compute the salience and save it to output folder
for k = 1:numel(fileList)
    
    fileName = fileList(k).name;
    fprintf('processed : %s\n', fileName);
    img = imread(fullfile(stimuli, fileName));
    
    saliency = compute_iseelSaliency(img, ensemble, nLearners, net, params);
    
    imwrite(saliency, [resultFolder, fileList(k).name]);
end