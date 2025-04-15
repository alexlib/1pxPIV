function [pivData] = pivSinglepixAnalyze(im1,im2,pivDataIn,pivPar)


% spStatus: meaning of bits
%     1 ... masked
%     2 ... Gaussian fit failed
%     3 ... detected as spurious
%     4 ... replaced
%     5 ... smoothed


pivData = pivDataIn;
% compute cross-correlation and auto-correlation functions
pivData = pivSinglepixCorrelate(im1,im2,pivData,pivPar);
pivData = pivSinglepixCorrelate(im1,im2,pivData,pivPar,'auto');

% fit cross- and auto-correlations by 2D gaussian function
pivData = pivSinglepixGaussFit(pivData,pivPar);
pivData = pivSinglepixGaussFitAC(pivData,pivPar);

% Do computations to evaluate 1pxPIV
pivData = pivSinglepixEvaluate(pivData,pivPar);

end