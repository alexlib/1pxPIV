function [Mean,StDev,Skew,Flat] = statmoment(X,dim)
% STATMOMENT - computes nth central statistical moment of X along dimension dim,
% ignoring all NaN's in X. X should have 1, 2 or 3 s´dimensions. Higher
% moments (n>2) are normalized by std(X).

if nargin<2
    dim = 1;
end

haveDimensions = size(X);
haveDimensions = numel(haveDimensions(haveDimensions>1));

switch haveDimensions
    case 1
        auxOK = ~isnan(X);
        X = X(auxOK);
        N = numel(X);
        Mean = mean(X);
        StDev = std(X);
        Skew = sum((X-Mean).^3)/N/StDev^3;
        Flat = sum((X-Mean).^4)/N/StDev^4;
    case 2
        if dim == 1
            X = X';
        elseif dim ~= 2
            fprintf('Error (statmoments.m): invalid dimension.\n');
            Mean = NaN;
            StDev = NaN;
            Skew = NaN;
            Flat = NaN;
            return;
        end
        auxOut = zeros(size(X,1),4);
        for kk = 1:size(X,1)
            [me,st,sk,fl] = statmoment(X(kk,:));
            auxOut(kk,:) = [me,st,sk,fl];
        end
        Mean = auxOut(:,1);
        StDev = auxOut(:,2);
        Skew = auxOut(:,3);
        Flat = auxOut(:,4);
    case 3
        switch dim
            case 1
                X = permute(X,[1,2,3]);
            case 2
                X = permute(X,[2,1,3]);
            case 3
                X = permute(X,[3,1,2]);
        end
        auxOut = zeros(size(X,2),size(X,3),4)+NaN;
        for kk = 1:size(X,2)
            for jj = 1:size(X,3)
                [me,st,sk,fl] = statmoment(X(:,kk,jj));
                auxOut(kk,jj,:) = [me,st,sk,fl];
            end
        end
        Mean = auxOut(:,:,1);
        StDev = auxOut(:,:,2);
        Skew = auxOut(:,:,3);
        Flat = auxOut(:,:,4);
end

