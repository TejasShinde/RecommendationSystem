clear all; close all;

load ('ratings');
%load ('movie_data');

observations = nnz(ratings);
hoIter = 10;
k = linspace(5, 95, hoIter);
tre = zeros(1, hoIter);
tse = zeros(1, hoIter);

num_samples = ceil(log2(observations));
tidx = find(ratings > 0);
tidx = tidx(randperm(length(tidx)));
%%%%%%%%%tidx = randperm(length(ratings(:)));
tidx = tidx(1:num_samples);
rateMatrix = ratings;
testMatrix = zeros(size(ratings));
testMatrix(tidx) = ratings(tidx);
rateMatrix(tidx) = 0;

% Global SVD Test:
lowRank = 175;
regularizer = 0.04;

[U, V] = myRecommender(rateMatrix, lowRank, regularizer);

predALS = U*V';

for i = 1:hoIter
    predknn = knnRecommender( rateMatrix, k(i) );
    
    lambda = 0.95;
    %Perform algorithm mixture
    pred = lambda * predALS + (1-lambda) * predknn;

    %find test mean
    n = size(testMatrix, 1);
    smean = mean(testMatrix);
    smean = smean ./ n;
    %perform post-processing
    pred = postProcess(pred, smean);
    
    trainRMSE = norm((pred - rateMatrix) .* (rateMatrix > 0), 'fro') / sqrt(nnz(rateMatrix > 0));
    testRMSE = norm((pred - testMatrix) .* (testMatrix > 0), 'fro') / sqrt(nnz(testMatrix > 0));

    tre(i) = trainRMSE; tse(i) = testRMSE;
    fprintf('Reg-%d\t%.4f\t%.4f\n', regularizer, trainRMSE, testRMSE);
    
end
