function [pivData] = pivSinglepixValidate(pivData,pivPar,noPass)

fprintf('Validating peak-fitting results...');
tic;

return;

% initialize fields
vlMedU = zeros(size(pivData.spU)) + NaN;       % velocity median in the neighborhood of given IA
vlMedV = vlMedU;
vlMedC = vlMedU;
vlRmsU = vlMedU;       % rms (from median) in the neighborhood of given IA
vlRmsV = vlMedU;
vlRmsC = vlMedU;
X = pivData.spX;
Y = pivData.spY;
U = pivData.spU;
V = pivData.spV;
Cx = pivData.spCx;
Cy = pivData.spCy;
CCmaxFit = pivData.spCCmaxFit;
Phi = pivData.spPhi;
status = pivData.spStatus;
if nargin<3, noPass = pivPar.spGfitNPasses; end

% get parameters with sipler names
distXY = pivPar.spVlDist(noPass);
passes = pivPar.spVlPasses(noPass);
tresh = pivPar.spVlTresh(noPass);
epsi = pivPar.spVlEps(noPass);
statusbit = 3;

for kpass = 1:passes       % proceed in two passes
    % replace anything invalid by NaN. Invalid vectors are those with any flag in status, except flags
    % "smoothed" or "smoothed in a sequence".
    auxStatus = status;
    auxReplace = logical(bitget(auxStatus,1)|bitget(auxStatus,2)|bitget(auxStatus,3));
    U(auxReplace) = NaN;                % replace anything masked, wrong, interpolated or spurious by NaN
    V(auxReplace) = NaN;
    Cx(auxReplace) = NaN;
    % pad U and V with NaNs to allow validation at borders
    auxU = padarray(U,[distXY, distXY],NaN);
    auxV = padarray(V,[distXY, distXY],NaN);
    auxC = padarray(Cx,[distXY, distXY],NaN);
    % validate inner cells
    for kx = 1:size(U,2)
        for ky = 1:size(U,1)
            % compute the medians and deviations from median
            auxNeighU = auxU(ky:ky+2*distXY,kx:kx+2*distXY);
            auxNeighV = auxV(ky:ky+2*distXY,kx:kx+2*distXY);
            auxNeighC = auxC(ky:ky+2*distXY,kx:kx+2*distXY);
            vlMedU(ky,kx) = median(removeNaNs(reshape(auxNeighU,1,(2*distXY+1)^2)));
            vlMedV(ky,kx) = median(removeNaNs(reshape(auxNeighV,1,(2*distXY+1)^2)));
            vlMedC(ky,kx) = median(removeNaNs(reshape(auxNeighC,1,(2*distXY+1)^2)));
            auxNeighU(distXY+1,distXY+1) = NaN;   % remove examined vector from the rms calculation
            auxNeighV(distXY+1,distXY+1) = NaN;
            auxNeighC(distXY+1,distXY+1) = NaN;
            vlRmsU(ky,kx) = stdfast(auxNeighU-vlMedU(ky,kx));  % rms of velues from the median
            vlRmsV(ky,kx) = stdfast(auxNeighV-vlMedV(ky,kx));
            vlRmsC(ky,kx) = stdfast(auxNeighC-vlMedC(ky,kx));
            if status(ky,kx) == 0 && abs(U(ky,kx)-vlMedU(ky,kx))>(tresh*vlRmsU(ky,kx)+epsi)
                status(ky,kx) = bitset(status(ky,kx),statusbit);
            end
            if status(ky,kx)==0 && abs(V(ky,kx)-vlMedV(ky,kx))>(tresh*vlRmsV(ky,kx)+epsi)
                status(ky,kx) = bitset(status(ky,kx),statusbit);
            end
            if status(ky,kx)==0 && abs(Cx(ky,kx)-vlMedC(ky,kx))>(tresh*vlRmsC(ky,kx)+epsi)
                status(ky,kx) = bitset(status(ky,kx),statusbit);
            end
        end
    end
end
% replace spurious vectors with NaN's
spurious = logical(bitget(status,statusbit));
U(spurious) = NaN;       
V(spurious) = NaN;
Cx(spurious) = NaN;
Cy(spurious) = NaN;
Phi(spurious) = NaN;
CCmaxFit(spurious) = NaN;

% output detailed pivData
auxSpur = logical(bitget(status,statusbit));
vlNSpur = sum(sum(auxSpur));    % number of spurious vectors
spuriousX = X(auxSpur);
spuriousY = Y(auxSpur);
spuriousU = pivData.spU(auxSpur);
spuriousV = pivData.spV(auxSpur);
spuriousCx = pivData.spCx(auxSpur);
spuriousCy = pivData.spCy(auxSpur);
spuriousPhi = pivData.spPhi(auxSpur);
spuriousCCmaxFit = pivData.spCCmaxFit(auxSpur);

% output variables
pivData.spU = U;
pivData.spV = V;
pivData.spCx = Cx;
pivData.spCy = Cy;
pivData.spPhi = Phi;
pivData.spCCmaxFit = CCmaxFit;
pivData.spStatus = uint16(status);
pivData.spSpuriousN = vlNSpur;

pivData.spSpuriousX = spuriousX;
pivData.spSpuriousY = spuriousY;
pivData.spSpuriousU = spuriousU;
pivData.spSpuriousV = spuriousV;
pivData.spSpuriousCx = spuriousCx;
pivData.spSpuriousCy = spuriousCy;
pivData.spSpuriousPhi = spuriousPhi;
pivData.spSpuriousCCmaxFit = spuriousCCmaxFit;

fprintf(' Finished in %.2fs.\n',toc);

end


%% XX Local functions

function [out] = removeNaNs(in)
% remove NaNs from a vector or column
out = in(~isnan(in));
end

function [out] = stdfast(in)
% computes root-mean-square (reprogramed, because std in Matlab is somewhat slow due to some additional tests)
in = reshape(in,1,numel(in));
notnan = ~isnan(in);
n = sum(notnan);
in(~notnan) = 0;
avg = sum(in)/n;
out = sqrt(sum(((in - avg).*notnan).^2)/(n-0)); % there should be -1 in the denominator for true std
end
