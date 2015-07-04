function [pivData] = pivSinglepixSmooth(pivData,pivPar,noPass)

if nargin<3, noPass = pivPar.spGFitNPasses; end

return;

Uin = double(pivData.spU);
Vin = double(pivData.spV);
Cxin = double(pivData.spCx);
Cyin = double(pivData.spCy);
Phiin = double(pivData.spPhi);
CCmaxFitin = double(pivData.spCCmaxFit);
status = pivData.spStatus;

sigma = pivPar.spSmSigma(noPass);

% perform smoothing only if sigma is not NaN
if isnan(sigma)
    return;
end

% following code is adapted from [3]
auxNanU = logical(isnan(Uin));  % remember NaNs and do not change them
auxNanV = logical(isnan(Vin));
auxNanCx = logical(isnan(Cxin));  % remember NaNs and do not change them
auxNanCy = logical(isnan(Cyin));
auxNanPhi = logical(isnan(Phiin));  % remember NaNs and do not change them
auxNanCCmaxFit = logical(isnan(CCmaxFitin));
if ~isnan(sigma)
    U = smoothn(Uin,sigma);
    V = smoothn(Vin,sigma);
    Cx = smoothn(Cxin,sigma);
    Cy = smoothn(Cyin,sigma);
    Phi = smoothn(Phiin,sigma);
    CCmaxFit = smoothn(CCmaxFitin,sigma);
else
    U = smoothn(Uin);
    V = smoothn(Vin);
    Cx = smoothn(Cxin);
    Cy = smoothn(Cyin);
    Phi = smoothn(Phiin);
    CCmaxFit = smoothn(CCmaxFitin);
end
U(auxNanU) = NaN;
V(auxNanV) = NaN;
Cx(auxNanCx) = NaN;
Cy(auxNanCy) = NaN;
Phi(auxNanPhi) = NaN;
CCmaxFit(auxNanCCmaxFit) = NaN;
status(~(auxNanU+auxNanV+auxNanCx+auxNanCy+auxNanPhi+auxNanCCmaxFit)) = ...
    bitset(status(~(auxNanU+auxNanV+auxNanCx+auxNanCy+auxNanPhi+auxNanCCmaxFit)),5);

% update PIV data
pivData.spU = U;
pivData.spV = V;
pivData.spCx = Cx;
pivData.spCy = Cy;
pivData.spPhi = Phi;
pivData.spCCmaxFit = CCmaxFit;

pivData.spStatus = uint16(status);