function [pivData] = pivSmooth(pivData,pivPar)
% pivSmooth - smooth displacement field in a set of PIV data
%
% Usage:
%     [pivData] = pivSmooth(pivData,pivPar)
%
% Inputs:
%     pivData ... (struct) structure containing more detailed results. Required field is
%        X, Y ... position, at which velocity/displacement is calculated
%        U, V ... displacements in x and y direction
%        Status ... matrix describing status of velocity vectors (for values, see Outputs section)
%     pivPar ... (struct) parameters defining the evaluation. Following fields are considered:
%                smMethod ... defines smoothing method. Possible values are:
%                    'none' ... do not perform smoothing
%                    'smoothn' ... uses smoothn.m function by Damian Garcia [5]
%                    'gauss' ... uses Gaussian kernel
%                smSigma ... amount of smoothing
%                smSize ... size of filter (applies only to Gaussian filter)
%
% Outputs:
%    pivData  ... (struct) structure containing more detailed results. If pivData was non-empty at the input, its
%              fields are preserved. Following field is modified:
%        U, V ... x and y components of the velocity/displacement vector (with replaced NaN's)
%        Status ... matrix with statuis of velocity vectors (uint8). Bits have this coding:
%            1 ... masked (set by pivInterrogate)
%            2 ... cross-correlation failed (set by pivCrossCorr)
%            4 ... peak detection failed (set by pivCrossCorr)
%            8 ... indicated as spurious by median test (set by pivValidate)
%           16 ... interpolated (set by pivReplaced)
%           32 ... smoothed (set by pivSmooth)
%
% Requirements:
%
% Credits:
%    This subroutine is a part of PIVsuite. This set of subroutines are a modified version of piv_FFTmulti.m
%    and other m-files, which form PIVlab software [3], developped by W. Thielicke and E. J. Stamhuis. In
%    PIVsuite, only a minor changes are done to the algorithms used in PIVlab, but the structure of
%    subroutines is different.
%
%    PIVsuite uses 3rd party software:
%        inpaint_nans.m, by J. D'Errico, [2]
%
% References:
%   [1] Adrian & Whesterweel, Particle Image Velocimetry, Cambridge University Press 2011
%   [2] John D'Errico, inpaint_nans subroutine, http://www.mathworks.com/matlabcentral/fileexchange/4551
%   [3] W. Thielicke and E. J. Stamhuid, PIVlab 1.31, http://pivlab.blogspot.com
%   [4] Raffel, Willert, Wereley & Kompenhans, Particle Image Velocimetry: A Practical Guide. 2nd edition,
%       Springer 2007
%   [5] Damien Garcia, smoothn subroutine, http://www.mathworks.com/matlabcentral/fileexchange/725-smoothn

issignle = isa(pivData.U,'single');

Uin = double(pivData.U);
Vin = double(pivData.V);
status = pivData.Status;
if size(Uin,3)>1
    method = pivPar.smMethodSeq;
    sigma = pivPar.smSigmaSeq;
    bit = 9;
else
    method = pivPar.smMethod;
    sigma = pivPar.smSigma;
    bit = 6;
end

switch lower(method)
    case 'none'
        U = Uin;
        V = Vin;
    case 'smoothn'
        % following code is adapted from [3]
        auxNanU = logical(isnan(Uin));  % remember NaNs and do not change them
        auxNanV = logical(isnan(Vin));
        if ~isnan(sigma)
            U = smoothn(Uin,sigma);
            V = smoothn(Vin,sigma);
        else
            U = smoothn(Uin);
            V = smoothn(Vin);
        end
        U(auxNanU) = NaN;
        V(auxNanV) = NaN;
        status(~(auxNanU+auxNanV)) = bitset(status(~(auxNanU+auxNanV)),bit);
    case 'gauss'
        if size(Uin,3)>1
            disp('Error (pivSmooth): smMethod "Gauss" is allowed only for data on image pair, not on a sequences.');
            return
        end
        % following code is taken from [3]
        h = fspecial('gaussian',pivPar.smSize,sigma);
        U = imfilter(Uin,h,'replicate');
        V = imfilter(Vin,h,'replicate');
        status(logical(~isnan(U)+~isnan(V))) = bitset(status(logical(~isnan(U)+~isnan(V))),bit);
        % restore elements, which became NaNs
        auxNanU = logical(isnan(U).*(~isnan(Uin)));
        auxNanV = logical(isnan(V).*(~isnan(Vin)));
        U(auxNanU) = Uin(auxNanU);
        V(auxNanV) = Vin(auxNanV);
    otherwise
        disp('warning: unknown smoothing method');
end

% update PIV data
if issignle
    pivData.U = single(U);
    pivData.V = single(V);
else
    pivData.U = double(U);
    pivData.V = double(V);
end
pivData.Status = uint16(status);






