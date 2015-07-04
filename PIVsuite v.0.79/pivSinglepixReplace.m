function [pivData] = pivSinglepixReplace(pivData,pivPar,noPass) %#ok<INUSD>

fprintf('Replacing spurious results...');
tic;

return;

% copy values
U = pivData.spU;
V = pivData.spV;
Cx = pivData.spCx;
Cy = pivData.spCy;
Phi = pivData.spPhi;
CCmaxFit = pivData.spCCmaxFit;
status = pivData.spStatus;

% detect all elements, for which replacement is required
containsNaN = logical(isnan(U)+isnan(V)+isnan(Cx)+isnan(Cy)+isnan(Phi)+isnan(CCmaxFit));
% detect all masked elements (replacement will be removed for them)
masked = logical(bitget(status,1));
replaced = logical(containsNaN .* (~masked));

% use D'Errico's inpaiting subroutine for 2D data for each time slice
U = inpaint_nans(U,4);
V = inpaint_nans(V,4);
Cx = inpaint_nans(Cx,4);
Cy = inpaint_nans(Cy,4);
Phi = inpaint_nans(Phi,4);
CCmaxFit = inpaint_nans(CCmaxFit,4);

% put back NaNs for masked elements
U(masked) = NaN;
V(masked) = NaN;
Cx(masked) = NaN;
Cy(masked) = NaN;
Phi(masked) = NaN;
CCmaxFit(masked) = NaN;
status(replaced) = bitset(status(replaced),4);

% output results
pivData.spU = U;
pivData.spV = V;
pivData.spCx = Cx;
pivData.spCy = Cy;
pivData.spPhi = Phi;
pivData.spCCmaxFit = CCmaxFit;
pivData.spStatus = uint16(status);

pivData.spReplacedN = sum(sum(replaced(:,:)));

fprintf(' Finished in %.2fs.\n',toc);