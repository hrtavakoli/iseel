

function neuralResponses = getNeuralResVGG16(net, img, scales, canonicalSize)

[~, ~, c] = size(img);
if c ~= 3
    img = repmat(img, [1,1,3]);
end    

img = single(img);

for i = 1:3
    img(:,:,i) = img(:,:,i) - net.normalization.averageImage(1,1,i);
end


neuralResponses = [];

for s = 1:numel(scales)
    
    img_c = imresize(img, scales{s}, 'bilinear');
    
    result = vl_simplenn(net, img_c);
    
    
    response = result(32).x;
    if size(response,1) ~= 18
        response = imresize(response, canonicalSize, 'bilinear');
    end
    neuralResponses = cat(3, neuralResponses, response);
    
end
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
neuralResponses = reshape(neuralResponses, [canonicalSize(1)*canonicalSize(2), size(neuralResponses, 3)]);
end

