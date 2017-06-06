

function [saliency] = compute_iseelSaliency(img, ensemble, nLearners, net, params)
% compute the iseel saliency model
%
% @input
%   img: input image
%   ensemble: a pool of learners
%   nLearners: number of learners for prediciton <= total leaners in the pool
%   net: is the vgg16 network to be used for computeing image features
%   params: some parameters of the network
%
% @output
%   saliency: predicted saliency map
%


% set post processing parameters
alpha = 6;
smoothingFactor = 13;


% load the center prior
load('center_prior');
center_prior = imresize(center_prior, params.resSize, 'bilinear');
center_prior = center_prior ./ sum(center_prior(:));

% this is a dummy input as we use the elm toolbox
fix = zeros(size(img,1),size(img,2));
fix = imresize(fix, params.resSize, 'bilinear');
fix = im2double(fix);
fix = reshape(fix, [params.resSize(1)*params.resSize(2), 1]);

% get the gist of the input
[gistDesc, objDesc] = getSceneDescription(net, img);
% get k top similar learners from the ensemble pool (490 from 700)
idxSim = getTopKSimilarImages(nLearners, ensemble, gistDesc, objDesc, 'ascend');
% project the input to deep feature space
neuralResponses = getNeuralResVGG16(net, img, params.scales, params.resSize);

output = zeros(params.resSize(1)*params.resSize(2), numel(idxSim));

for i = 1:numel(idxSim)
    cEnsemble = ensemble(idxSim(i));
    
    feature = bsxfun(@rdivide, neuralResponses, sqrt(sum(neuralResponses.^2, 2)));
    output(:, i) = elm_predict(cEnsemble.elmNet, feature, fix);
end

output = tanh(output);
output = sum(output, 2);
output(output < 0) = 0;
output = output./sum(output(:));

output = reshape(output.^alpha, params.resSize);
output = output.*center_prior;

[h, w, c] = size(img);
output = imresize(output, [h, w], 'bilinear');
output = imfilter(output, fspecial('gaussian', round([smoothingFactor*4, smoothingFactor*4]), smoothingFactor), 'replicate');
saliency = output / max(output(:));
