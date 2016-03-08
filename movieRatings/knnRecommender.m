function [ prediction ] = knnRecommender( rateMatrix, k )
%KNNRECOMMENDER : Finds recommendation using k nearest neighbors
%Nearness is defined using cosine measure

scheme = 'cosine';
dist = pdist2(rateMatrix, rateMatrix, scheme);
[sdist sidx] = sort(dist, 2);
prediction = rateMatrix;

for u = 1:size(rateMatrix, 1)
    neighbors = sidx(u, 1:k);
    avgNeighbor = mean(rateMatrix(neighbors, :));
    zeroRatingIdx = rateMatrix(u, :)==0;
    prediction(u, zeroRatingIdx) = avgNeighbor(zeroRatingIdx);
end

end

