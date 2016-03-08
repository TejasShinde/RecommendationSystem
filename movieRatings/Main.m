clear all; close all;

load ('ratings');

observations = nnz(ratings);
hoIter = 1;

for i = 1:hoIter
    num_samples = ceil(log2(observations));
    tidx = find(ratings > 0);
    tidx = tidx(randperm(length(tidx)));
    %tidx = randperm(length(ratings(:)));
    tidx = tidx(1:num_samples);
    rateMatrix = ratings;
    testMatrix = zeros(size(ratings));
    testMatrix(tidx) = ratings(tidx);
    rateMatrix(tidx) = 0;

    % Global SVD Test:
    lowRank = [1, 7, 13, 25, 50];
    llr = length(lowRank);
    regularizer = logspace(-3, 1, 5);
    lreg = length(regularizer);
    tre = zeros(llr, lreg);
    tse = zeros(llr, lreg);
    start = cputime;
    for l=1:llr
        for r = 1:lreg
            [U, V] = myRecommender(rateMatrix, lowRank(l), regularizer(r));

            pred = U*V';
            %pred = postProcess(pred);
            trainRMSE = norm((pred - rateMatrix) .* (rateMatrix > 0), 'fro') / sqrt(nnz(rateMatrix > 0));

            %find test mean
            n = size(testMatrix, 1);
            smean = mean(testMatrix);
            smean = smean ./ n;
            %perform post-processing
            pred = postProcess(pred, smean);
            testRMSE = norm((pred - testMatrix) .* (testMatrix > 0), 'fro') / sqrt(nnz(testMatrix > 0));

            fprintf('SVD-%d\t%.4f\t%.4f\n', lowRank(l), trainRMSE, testRMSE);
            tre(l,r) = trainRMSE; tse(l,r) = testRMSE;
        end
    end

    figure(2*i);
    surf(tre);
    figure(2*i+1);
    surf(tse);
%    plot(tre, 'red');
%    hold on
%    plot(tse, 'blue');
%    hold off
end
