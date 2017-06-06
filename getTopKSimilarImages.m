

function idx = getTopKSimilarImages(k, ensemble, gistDesc, objDesc, sortType)
% get the index to top k similar images from the ensemble to current image

distance = zeros(numel(ensemble), 1);
%objDistance = gistDistance;
for i = 1:numel(ensemble)
    cEnsemble = ensemble(i);
 
    distance(i) = sqrt(sum(([cEnsemble.gistDesc cEnsemble.objDesc] - [gistDesc objDesc]).^2));
end

[~, idx] = sort(distance, sortType);

idx = idx(1:k);

