%Name: Shinde Tejas S
%GTID: 902837071

function [ U, V ] = myRecommender( rateMatrix, lowRank , regularizer )
    % Parameters
    maxIter = 75; % Choose your own.
    learningRate = 5e-3; % Choose your own.
    %regularizer = 5e-2; % Choose your own.
    
    % Random initialization:
    [n1, n2] = size(rateMatrix);
    U = rand(n1, lowRank) / lowRank;
    V = rand(n2, lowRank) / lowRank;

    % Gradient Descent:
    for iter = 1:maxIter
        for u = 1:n1
            for i = 1:n2
                if rateMatrix(u,i)>0
        		    Eui = rateMatrix(u,i) - U(u,:) * V(i,:)';
                    U(u,:) = U(u,:) + learningRate * 2 * (Eui * V(i,:) - regularizer * U(u,:));
                    V(i,:) = V(i,:) + learningRate * 2 * (Eui * U(u,:) - regularizer * V(i,:));
                end
            end
        end
        e = 0;
        for u = 1:n1
            for i = 1:n2
                if rateMatrix(u,i)>0
                    e = e + (rateMatrix(u,i) - U(u,:) * V(i,:)')^2;
                    e = e + sum(regularizer * (U(u,:).^2 + V(i,:).^2));
                end
            end
        end
        if e < 1e-3
            %fprintf('converged');
            break;
        end
    end
end

