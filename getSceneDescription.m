


function [gistDesc, objDesc] = getSceneDescription(net, img)

    % compute the gist
    
    % we use the default gist parameters
    param.imageSize = [256 256]; % it works also with non-square images
    param.orientationsPerScale = [8 8 8 8];
    param.numberBlocks = 4;
    param.fc_prefilt = 4;

    % Computing gist energies
    gistDesc = LMgist(img, '', param);    

    [~, ~, c] = size(img);
    if c ~= 3
        img = repmat(img, [1,1,3]);
    end    

    img = single(img);

    img_c = imresize(img, [224 224], 'bilinear');
    
    img_c = img_c - net.normalization.averageImage;    
    
    result = vl_simplenn(net, img_c);

    objDesc = squeeze(result(end).x)';
    
    
end
