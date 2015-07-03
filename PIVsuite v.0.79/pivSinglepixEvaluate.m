function [pivData] = pivSinglepixEvaluate(pivDataIn,pivPar)

fprintf('Processing results of 1Px correlation results...\n');

% initialize fields and variables
pivData = pivDataIn;

nx = size(pivData.spCC,2);
ny = size(pivData.spCC,1);
dX = pivData.spX(1,2)-pivData.spX(1,1);
dY = pivData.spY(2,1)-pivData.spY(1,1);

aux = zeros(ny,nx) + NaN;
if ~isfield(pivData,'spD'), pivData.spD = aux; end
if ~isfield(pivData,'spC1'), pivData.spC1 = aux; end
if ~isfield(pivData,'spC2'), pivData.spC2 = aux; end
if ~isfield(pivData,'spP1'), pivData.spC1 = aux; end
if ~isfield(pivData,'spP2'), pivData.spC2 = aux; end
if ~isfield(pivData,'spRSuuRaw'), pivData.spRSuuRaw = aux; end
if ~isfield(pivData,'spRSvvRaw'), pivData.spRSvvRaw = aux; end
if ~isfield(pivData,'spRSuvRaw'), pivData.spRSuvRaw = aux; end
if ~isfield(pivData,'spRSuu'), pivData.spRSuu = aux; end
if ~isfield(pivData,'spRSvv'), pivData.spRSvv = aux; end
if ~isfield(pivData,'spRSuv'), pivData.spRSuv = aux; end
if ~isfield(pivData,'spUfiltered'), pivData.spUfiltered = aux; end
if ~isfield(pivData,'spVfiltered'), pivData.spVfiltered = aux; end
if ~isfield(pivData,'spdUdX'), pivData.spdUdX = aux; end
if ~isfield(pivData,'spdUdY'), pivData.spdUdY = aux; end
if ~isfield(pivData,'spdVdX'), pivData.spdVdX = aux; end
if ~isfield(pivData,'spdVdY'), pivData.spdVdY = aux; end

% compute size of particles (Scharnowski, Exp Fluids 52(2012):985-1002, p. 991, second paragraph and Fig. 2)
pivData.spD = (pivData.spACC1fit+pivData.spACC2fit)/(2*sqrt(2));
pivData.spD = -3.08*exp(-1.46*pivData.spD)-0.337+1.057*pivData.spD;

% correct C1 and C2 (Scharnowski, Exp Fluids 52(2012):985-1002, p. 991, fifth paragraph and Fig. 3; correction
% applied to both C1 and C2)
pivData.spC1 = -2.67*exp(-0.81*pivData.spC1fit)-0.393+1.043*pivData.spC1fit;
pivData.spC2 = -2.67*exp(-0.81*pivData.spC2fit)-0.393+1.043*pivData.spC2fit;

% compute P1 and P2 (Scharnowski, Exp Fluids 52(2012):985-1002, using eq. 13, see also last paragraph of 
% section 3.1)
% ################ Some Fail flag should be here when D > C1 or C2 ########################
pivData.spP1 = pivData.spC1.^2-2*pivData.spD.^2;
pivData.spP2 = pivData.spC2.^2-2*pivData.spD.^2;
auxNOK = (pivData.spP1<=0);
pivData.spP1(auxNOK) = NaN;
auxNOK = (pivData.spP2<=0);
pivData.spP2(auxNOK) = NaN;
pivData.spP1 = sqrt(pivData.spP1);
pivData.spP2 = sqrt(pivData.spP2);

% compute Reynolds stresses (Scharnowski, Exp Fluids 52(2012):985-1002, using eq. 14, see also last paragraph of 
% section 3.1)
pivData.spRSuuRaw = 1/16 * (cos(pivData.spPhifit).^2.*pivData.spP1.^2 + sin(pivData.spPhifit).^2.*pivData.spP2.^2);
pivData.spRSvvRaw = 1/16 * (sin(pivData.spPhifit).^2.*pivData.spP1.^2 + cos(pivData.spPhifit).^2.*pivData.spP2.^2);
pivData.spRSuvRaw = 1/16 * sin(pivData.spPhifit).*cos(pivData.spPhifit).*(pivData.spP2.^2 -pivData.spP1.^2);

% filter velocity fields (Adrian, eq. (9.11) on p. 432)
% U field
aux = pivData.spUfit;
aux = [aux(:,1) aux aux(:,end)];
pivData.spUfiltered = 0.5*aux(:,2:end-1) + 0.25*aux(:,1:end-2) + 0.25*aux(:,3:end);
% V field
aux = pivData.spVfit;
aux = [aux(1,:); aux; aux(end,:)];
pivData.spVfiltered = 0.5*aux(2:end-1,:) + 0.25*aux(1:end-2,:) + 0.25*aux(3:end,:);
% filter also in other direction
% U field
aux = pivData.spUfiltered;
aux = [aux(1,:); aux; aux(end,:)];
pivData.spUfiltered2 = 0.5*aux(2:end-1,:) + 0.25*aux(1:end-2,:) + 0.25*aux(3:end,:);
% V field
aux = pivData.spVfiltered;
aux = [aux(:,1) aux aux(:,end)];
pivData.spVfiltered2 = 0.5*aux(:,2:end-1) + 0.25*aux(:,1:end-2) + 0.25*aux(:,3:end);

% compute the derivatives
% derivatives of U
aux = zeros(ny+2,nx+2) + NaN;
aux(2:end-1,2:end-1) = pivData.spUfiltered2;
pivData.spdUdX = (aux(2:end-1,3:end)-aux(2:end-1,1:end-2))/(2*dX);
pivData.spdUdY = (aux(3:end,2:end-1)-aux(1:end-2,2:end-1))/(2*dY);
% derivatives of V
aux = zeros(ny+2,nx+2) + NaN;
aux(2:end-1,2:end-1) = pivData.spVfiltered2;
pivData.spdVdX = (aux(2:end-1,3:end)-aux(2:end-1,1:end-2))/(2*dX);
pivData.spdVdY = (aux(3:end,2:end-1)-aux(1:end-2,2:end-1))/(2*dY);

% correct the Reynolds stresses for the velocity gradients
pivData.spRSuu = pivData.spRSuuRaw - 1/16 * (pivData.spD.^2) .* (pivData.spdUdY.^2);
pivData.spRSvv = pivData.spRSvvRaw - 1/16 * (pivData.spD.^2) .* (pivData.spdVdX.^2);
pivData.spRSuv = pivData.spRSuvRaw - 1/16 * (pivData.spD.^2) .* (pivData.spdUdY+pivData.spdVdX);

