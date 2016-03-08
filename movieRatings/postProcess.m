function [ processed ] = postProcess( unprocessed, smean )
%POSTPROCESS Summary of this function goes here
%   Detailed explanation goes here

%1. Item Based Correction
n = size(unprocessed, 1);
spmean = mean(unprocessed);
spmean = spmean ./ n;
correction = smean - spmean;
for mm = 1:length(correction)
    unprocessed(:,mm) = unprocessed(:,mm) + correction(mm);
end

%2. Prediction Truncation
underflow = unprocessed < 1;
overflow = unprocessed > 5;

processed = unprocessed;
processed(underflow) = 1;
processed(overflow) = 5;

%3. Near Integer Round-off
for i = 1:5
    idxi = abs(processed - i) < .1;
    processed(idxi) = i;
end

end

