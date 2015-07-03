function [pivData,ccPeakIm] = pivCrossCorr(exIm1,exIm2,pivData,pivPar)
% pivCrossCorr - cross-correlates two images to find object displacement between them
%
% Usage:
% [X,Y,U,V,pivData,ccPeakIm] = pivCrossCorr(exIm1,exIm2,pivData,pivPar)
%
% Inputs:
%    Exim1,Exim2 ... pair of expanded images (use pivInterrogate for their generation)
%    pivData ... (struct) structure containing more detailed results. Following fields are required (use
%             pivInterrogate for generating them):
%        Status ... matrix describing status of velocity vectors (for values, see Outputs section)
%        iaX, iaY ... matrices with centers of interrogation areas
%        iaU0, iaV0 ... mean shift of IA's
%    pivPar ... (struct) parameters defining the evaluation. Following fields are considered:
%       ccRemoveIAMean ... if =0, do not remove IA's mean before cross-correlation; if =1, remove the mean; if
%                          in between, remove the mean partially
%       ccMaxDisplacement ... maximum allowed displacement to accept cross-correlation peak. This parameter is 
%                             a multiplier; e.g. setting ccMaxDisplacement = 0.6 means that the cross-
%                             correlation peak must be located within [-0.6*iaSizeX...0.6*iaSizeX,
%                             -0.6*iaSizeY...0.6*iaSizeY] from the zero displacement. Note: IA offset is not 
%                             included in the displacement, hence real displacement can be larger if 
%                             ccIAmethod is any other than 'basic'.
%
% Outputs:
%    pivData  ... (struct) structure containing more detailed results. Following fields are added or updated:
%        X, Y, U, V ... contains velocity field
%        Status ... matrix with statuis of velocity vectors (uint8). Bits have this coding:
%            1 ... masked (set by pivInterrogate)
%            2 ... cross-correlation failed (set by pivCrossCorr)
%            4 ... peak detection failed (set by pivCrossCorr)
%            8 ... indicated as spurious by median test (set by pivValidate)
%           16 ... interpolated (set by pivReplaced)
%           32 ... smoothed (set by pivSmooth)
%        ccPeak ... table with values of cross-correlation peak
%        ccPeakSecondary ... table with values of secondary cross-correlation peak (maximum of
%            crosscorrelation, if 5x5 neighborhood of primary peak is removed)
%        ccStd1, ccStd2 ... tables with standard deviation of pixel values in interrogation area, for the
%            first and second image in the image pair
%        ccMean1, ccMean2 ... tables with mean of pixel values in interrogation area, for the
%            first and second image in the image pair
%        ccFailedN ... number of vectors for which cross-correlation failed
%            at distance larger than ccMaxDisplacement*(iaSizeX,iaSizeY) )
%        ccSubpxFailedN ... number of vectors for which subpixel interpolation failed
%      - Note: fields iaU0 and iaV0 are removed from pivData
%    ccPeakIm ... expanded image containing cross-correlation functions for all IAs (normalized by .ccPeak)
% 
% Important local variables:
%    failFlag ... contains value of status elements of the vector being processed
%    Upx, Vpx ... rough position of cross-correlation peak (before subpixel interpolation, in integer 
%                     number of pixels)
%
%        
% This subroutine is a part of
%
% =========================================
%               PIVsuite
% =========================================
%
% PIVsuite is a set of subroutines intended for processing of data acquired with PIV (particle image
% velocimetry) within Matlab environment. 
%
% Written by Jiri Vejrazka, Institute of Chemical Process Fundamentals, Prague, Czech Republic
%
% For the use, see files example_XX_xxxxxx.m, which acompany this file. PIVsuite was tested with
% Matlab 7.12 (R2011a) and 7.14 (R2012a).
%
% In the case of a bug, please, contact me: vejrazka (at) icpf (dot) cas (dot) cz
%
%
% Requirements:
%     Image Processing Toolbox
% 
%     inpaint_nans.m
%         subroutine by John D'Errico, available at http://www.mathworks.com/matlabcentral/fileexchange/4551
%
%     smoothn.m
%         subroutine by Damien Garcia, available at 
%         http://www.mathworks.com/matlabcentral/fileexchange/274-smooth
%
% Credits:
%    PIVsuite is a redesigned version of PIVlab software [3], developped by W. Thielicke and E. J. Stamhuis. 
%    Some parts of this code are copied or adapted from it (especially from its piv_FFTmulti.m subroutine). 
%    PIVsuite uses 3rd party software:
%        inpaint_nans.m, by J. D'Errico, [2]
%        smoothn.m, by Damien Garcia, [5]
%        
% References:
%   [1] Adrian & Whesterweel, Particle Image Velocimetry, Cambridge University Press 2011
%   [2] John D'Errico, inpaint_nans subroutine, http://www.mathworks.com/matlabcentral/fileexchange/4551
%   [3] W. Thielicke and E. J. Stamhuid, PIVlab 1.31, http://pivlab.blogspot.com
%   [4] Raffel, Willert, Wereley & Kompenhans, Particle Image Velocimetry: A Practical Guide. 2nd edition,
%       Springer 2007
%   [5] Damien Garcia, smoothn subroutine, http://www.mathworks.com/matlabcentral/fileexchange/274-smooth


% Acronyms and meaning of variables used in this subroutine:
%    IA ... concerns "Interrogation Area"
%    im ... image
%    dx ... some index
%    ex ... expanded (image)
%    est ... estimate (velocity from previous pass) - will be used to deform image
%    aux ... auxiliary variable (which is of no use just a few lines below)
%    cc ... cross-correlation
%    vl ... validation
%    sm ... smoothing
%    Word "velocity" should be understood as "displacement"



%% 0. Initialization

% X = pivData.X;
% Y = pivData.Y;
U = pivData.U;
V = pivData.V;
status = pivData.Status;
ccPeak = U;                  % will contain peak levels
ccPeakSecondary = U;         % will contain level of secondary peaks
iaSizeX = pivPar.iaSizeX;
iaSizeY = pivPar.iaSizeY;
iaNX = size(pivData.X,2);
iaNY = size(pivData.X,1);
ccStd1 = pivData.U+NaN;
ccStd2 = pivData.U+NaN;
ccMean1 = pivData.U+NaN;
ccMean2 = pivData.U+NaN;


% initialize "expanded image" for storing cross-correlations
ccPeakIm = exIm1 + NaN;   % same size as expanded images

% peak position is shifted by 1 or 0.5 px, depending on IA size
if rem(iaSizeX,2) == 0
    ccPxShiftX = 1;
else
    ccPxShiftX = 0.5;
end
if rem(iaSizeY,2) == 0
    ccPxShiftY = 1;
else
    ccPxShiftY = 0.5;
end


%% 1. Cross-correlate expanded images and do subpixel interpolation

% loop over interrogation areas
for kx = 1:iaNX
    for ky = 1:iaNY
        failFlag = status(ky,kx);
        % if not masked, get individual interrogation areas from the expanded images
        if failFlag == 0
            imIA1 = exIm1(1+(ky-1)*iaSizeY:ky*iaSizeY,1+(kx-1)*iaSizeX:kx*iaSizeX);
            imIA2 = exIm2(1+(ky-1)*iaSizeY:ky*iaSizeY,1+(kx-1)*iaSizeX:kx*iaSizeX);
            % remove IA mean
            auxMean1 = mean2(imIA1);
            auxMean2 = mean2(imIA2);
            imIA1 = imIA1 - pivPar.ccRemoveIAMean*auxMean1;
            imIA2 = imIA2 - pivPar.ccRemoveIAMean*auxMean2;
            % compute rms for normalization of cross-correletion
            auxStd1 = stdfast(imIA1);
            auxStd2 = stdfast(imIA2);
            % do the cross-correlation and normalize it
            cc = fftshift(real(ifft2(conj(fft2(imIA1)).*fft2(imIA2))))/(auxStd1*auxStd2)/(iaSizeX*iaSizeY);
            % find the cross-correlation peak
            [auxPeak,Upx] = max(max(cc));
            [aux,Vpx] = max(cc(:,Upx));     %#ok<ASGLU>
            % if the displacement is too large (too close to border), set fail flag
            if (abs(Upx-iaSizeX/2-ccPxShiftX) > pivPar.ccMaxDisplacement*iaSizeX) || ...
                    (abs(Vpx-iaSizeY/2-ccPxShiftY) > pivPar.ccMaxDisplacement*iaSizeY)
                failFlag =  bitset(failFlag,2);
            end
            % sub-pixel interpolation (2x3point Gaussian fit, eq. 8.163, p. 375 in [1])
            try
                dU = (log(cc(Vpx,Upx-1)) - log(cc(Vpx,Upx+1)))/...
                    (log(cc(Vpx,Upx-1))+log(cc(Vpx,Upx+1))-2*log(cc(Vpx,Upx)))/2;
                dV = (log(cc(Vpx-1,Upx)) - log(cc(Vpx+1,Upx)))/...
                    (log(cc(Vpx-1,Upx))+log(cc(Vpx+1,Upx))-2*log(cc(Vpx,Upx)))/2;
            catch     %#ok<*CTCH>
                failFlag = bitset(failFlag,3);
                dU = NaN; dV = NaN;
            end
            % if imaginary, set fail flag
            if (~isreal(dU)) || (~isreal(dV))
                failFlag = bitset(failFlag,3);
            end
        else
            cc = zeros(iaSizeY,iaSizeX) + NaN;
            auxPeak = NaN;            
            auxStd1 = NaN;            
            auxStd2 = NaN;
            auxMean1 = NaN;            
            auxMean2 = NaN;
            Upx = iaSizeX/2;
            Vpx = iaSizeY/2;
        end
        % save the pivData information about cross-correlation, rough peak position and peak level
        if failFlag == 0
            U(ky,kx) = pivData.iaU0(ky,kx) + Upx + dU - iaSizeX/2 - ccPxShiftX;               % this is subroutine's output
            V(ky,kx) = pivData.iaV0(ky,kx) + Vpx + dV - iaSizeY/2 - ccPxShiftY;               % this is subroutine's output
        else
            U(ky,kx) = NaN;
            V(ky,kx) = NaN;
        end
        status(ky,kx) = failFlag;
        ccPeakIm(1+(ky-1)*iaSizeY:ky*iaSizeY,1+(kx-1)*iaSizeX:kx*iaSizeX) = cc;
        ccPeak(ky,kx) = auxPeak;
        ccStd1(ky,kx) = auxStd1;
        ccStd2(ky,kx) = auxStd2;
        ccMean1(ky,kx) = auxMean1;
        ccMean2(ky,kx) = auxMean2;
        % find secondary peak
        try
            cc(Vpx-2:Vpx+2,Upx-2:Upx+2) = 0;
            ccPeakSecondary(ky,kx) = max(max(cc));
        catch
            try    
                cc(Vpx-1:Vpx+1,Upx-1:Upx+1) = 0;
                ccPeakSecondary(ky,kx) = max(max(cc));
            catch
                ccPeakSecondary(ky,kx) = NaN;
            end
        end % end of secondary peak search
    end % end of loop for ky
end % end of loop for kx

% get IAs where CC failed, and coordinates of corresponding IAs
ccFailedI = logical(bitget(status,2));
ccSubpxFailedI = logical(bitget(status,3));


%% 2. Output results
pivData.Status = uint16(status);
pivData.U = U;
pivData.V = V;
pivData.ccPeak = ccPeak;
pivData.ccPeakSecondary = ccPeakSecondary;
pivData.ccStd1 = ccStd1;
pivData.ccStd2 = ccStd2;
pivData.ccMean1 = ccMean1;
pivData.ccMean2 = ccMean2;
pivData.ccFailedN = sum(sum(ccFailedI));
pivData.ccSubpxFailedN = sum(sum(ccSubpxFailedI));
pivData = rmfield(pivData,'iaU0');
pivData = rmfield(pivData,'iaV0');

end


%% LOCAL FUNCTIONS

function [out] = stdfast(in)
% computes root-mean-square (reprogramed, because std in Matlab is somewhat slow due to some additional tests)
in = reshape(in,1,numel(in));
notnan = ~isnan(in);
n = sum(notnan);
in(~notnan) = 0;
avg = sum(in)/n;
out = sqrt(sum(((in - avg).*notnan).^2)/(n-0)); % there should be -1 in the denominator for true std
end
