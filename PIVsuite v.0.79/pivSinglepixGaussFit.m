function [pivData] = pivSinglepixGaussFit(pivDataIn,pivPar)

fprintf('Fitting cross-correlation peaks by 2D gaussian function...\n');

% initialize fields and variables

pivData = pivDataIn;
nx = size(pivData.spCC,2);
ny = size(pivData.spCC,1);
aux = zeros(ny,nx) + NaN;
if ~isfield(pivData,'spU0'), pivData.spU0 = aux; end
if ~isfield(pivData,'spV0'), pivData.spV0 = aux; end
if ~isfield(pivData,'spUint'), pivData.spUint = aux; end
if ~isfield(pivData,'spVint'), pivData.spVint = aux; end
if ~isfield(pivData,'spUfit'), pivData.spUfit = aux; end
if ~isfield(pivData,'spVfit'), pivData.spVfit = aux; end
if ~isfield(pivData,'spStatus'), pivData.spStatus = uint16(zeros(ny,nx)); end
if ~isfield(pivData,'spC0int'), pivData.spC0int = aux; end
if ~isfield(pivData,'spC1int'), pivData.spC1int = aux; end
if ~isfield(pivData,'spC2int'), pivData.spC2int = aux; end
if ~isfield(pivData,'spC0fit'), pivData.spC0fit = aux; end
if ~isfield(pivData,'spC1fit'), pivData.spC1fit = aux; end
if ~isfield(pivData,'spC2fit'), pivData.spC2fit = aux; end
if ~isfield(pivData,'spPhiint'), pivData.spPhiint = aux; end
if ~isfield(pivData,'spPhifit'), pivData.spPhifit = aux; end
if ~isfield(pivData,'spCCmax'), pivData.spCCmax = aux; end
if ~isfield(pivData,'spOFfit'), pivData.spOFfit = aux; end
if ~isfield(pivData,'spOFint'), pivData.spOFint = aux; end
if isfield(pivPar,'spEchoInt')
    spEchoInt = pivPar.spEchoInt;
else
    spEchoInt = 20;
end

% clear "fit failed", "invalid", "replaced" and "smoothed" flag
aux = pivData.spStatus;
aux = bitset(aux,2,0);
aux = bitset(aux,3,0);
aux = bitset(aux,4,0);
aux = bitset(aux,5,0);
pivData.spStatus = aux;

% check the existance of a file with intermediate results
prefix = ['spCC_GaussFit_' num2str(pivPar.spBindX,'%d') 'x' num2str(pivPar.spBindY,'%d')];
if pivPar.spBindX~=pivPar.spStepX || pivPar.spBindY~=pivPar.spStepY
     prefix = [prefix 'step' num2str(pivPar.spStepX,'%d') 'x' num2str(pivPar.spStepY,'%d')];
end
prefix = [prefix '_'];
fname1 = pivDataIn.spImg1First;
fname2 = pivDataIn.spImg2First;
fname3 = pivDataIn.spImg1Second;
fname4 = pivDataIn.spImg2Last;
if pivPar.spOnDrive && ~pivPar.spForceProcessing
    % get the kx index for the file, in which results are stored
    aux = dir([pivPar.spTargetPath,filesep,prefix,fname1,'_',fname2,'_',fname3,'_',fname4,'_*.mat']);
    auxfiles = cell(numel(aux,1));
    for ki = 1:numel(aux)
        auxfiles{ki,1} = aux(ki).name;
    end
    if numel(aux)>0
        auxfiles = sort(auxfiles);
        LastFile = auxfiles{end};
        auxCharsToRemove = numel([prefix,fname1,'_',fname2,'_',fname3,'_',fname4])+2;
        kxStart = str2double(LastFile(auxCharsToRemove:end-4))+1;
        fprintf('    Reading results of CC peak fit from %s...', LastFile);
        tic;
        aux = load([pivPar.spTargetPath,filesep,LastFile]);
        pivData.spU0 = aux.spU0;
        pivData.spV0 = aux.spV0;
        pivData.spUint = aux.spUint;
        pivData.spVint = aux.spVint;
        pivData.spUfit = aux.spUfit;
        pivData.spVfit = aux.spVfit;
        pivData.spC0int = aux.spC0int;
        pivData.spC1int = aux.spC1int;
        pivData.spC2int = aux.spC2int;
        pivData.spC0fit = aux.spC0fit;
        pivData.spC1fit = aux.spC1fit;
        pivData.spC2fit = aux.spC2fit;
        pivData.spPhiint = aux.spPhiint;
        pivData.spPhifit = aux.spPhifit;
        pivData.spCCmax = aux.spCCmax;
        pivData.spStatus = aux.spStatus;
        pivData.spOFfit = aux.spOFfit;
        pivData.spOFint = aux.spOFint;
        fprintf(' Finished in %.2fs.\n',toc);
    else
        kxStart = 1;
        LastFile = '';
    end
else
    kxStart = 1;
    LastFile = '';
end

% loop over all coordinates
t1 = tic;
auxCount = (kxStart-1)*ny;
for kx = kxStart:nx
    for ky = 1:ny
        % 1a. if pixel is masked, skip it
        if bitget(pivData.spStatus(ky,kx),1)
            continue
        end
        % 1b. give some echo
        if (auxCount/spEchoInt == round(auxCount/spEchoInt))&&(auxCount>0)
            fprintf('    Fitting cross-correlation peak for velocity vector %d of %d...',auxCount,nx*ny);
            t2 = tic;
        end
        cc = squeeze(pivDataIn.spCC(ky,kx,:,:));
        % 2.  SUBPIXEL INTERPOLATION TO FIND MAXIMA (similar to standard PIV)
        % 2a. obtain initial guess for the CC peak
        [~,Upx] = max(max(cc));    % guess based on maxima of CC
        [CCmax,Vpx] = max(cc(:,Upx));
        % 2b. if maxima is on border of CC, set "fit failed" flag
        if Upx <= 1 || Vpx <= 1 || Upx >= size(cc,2)-1 || Vpx >= size(cc,1)-1
            pivData.spStatus(ky,kx)=bitset(pivData.spStatus(ky,kx),2,1);
        end
        % 2c. find initial guess of (U0,V0) based on 2x3 point gaussian fit
        if pivData.spStatus(ky,kx)==0
            y11 = log(cc(Vpx,Upx-1));
            y12 = log(cc(Vpx,Upx));
            y13 = log(cc(Vpx,Upx+1));
            y21 = log(cc(Vpx-1,Upx));
            y22 = log(cc(Vpx,Upx));
            y23 = log(cc(Vpx+1,Upx));
            U0 = (y11 - y13)/(y11+y13-2*y12)/2;
            V0 = (y21 - y23)/(y21+y23-2*y22)/2;
            Upx = Upx - pivPar.spDeltaXNeg - 1;
            Vpx = Vpx - pivPar.spDeltaYNeg - 1;
            U0 = Upx + U0;
            V0 = Vpx + V0;
            CCmax = exp(y12 -(y11-y13)^2/(y11+y13-2*y12)/8 -(y21-y23)^2/(y21+y23-2*y22)/8);
        else
            Upx = Upx - pivPar.spDeltaXNeg - 1;
            Vpx = Vpx - pivPar.spDeltaYNeg - 1;
            U0 = Upx;
            V0 = Vpx;
        end
        % 2d. if fails, use only maximum with integer-pixel accuracy
        if ~isreal(U0) ||~isreal(V0)
            U0 = Upx;
            V0 = Vpx;
        end
        % 3.  COMPUTATION OF PEAK PARAMETERS USING INTEGRAL VALUES OF CC
        % 3a. create matrices with coordinates
        X = ones(size(cc,1),1) * (-pivPar.spDeltaXNeg:pivPar.spDeltaXPos);
        Y = (-pivPar.spDeltaYNeg:pivPar.spDeltaYPos)' * ones(1,size(cc,2));
        % 3b. compute integrals
        I0 = sum(sum(cc));
        Uint = sum(sum(X.*cc))/I0;
        Vint = sum(sum(Y.*cc))/I0;
        % Iuu =  16/I0 * sum(sum((X-Uint).^2.*cc));
        % Ivv =  16/I0 * sum(sum((Y-Vint).^2.*cc));
        % Iuv = -32/I0 * sum(sum((X-Uint).*(Y-Vint).*cc));
        Iuu =  16/I0 * sum(sum((X-U0).^2.*cc));
        Ivv =  16/I0 * sum(sum((Y-V0).^2.*cc));
        Iuv = -32/I0 * sum(sum((X-U0).*(Y-V0).*cc));
        % 3c. compute parameters of Gaussian peak
        C1int = abs(sqrt((Iuu^2-Ivv^2-sqrt((Iuu-Ivv)^2*(Iuu^2+Ivv^2+Iuv^2-2*Iuu*Ivv)))/(2*(Iuu-Ivv))));
        C2int = abs(sqrt((Iuu^2-Ivv^2+sqrt((Iuu-Ivv)^2*(Iuu^2+Ivv^2+Iuv^2-2*Iuu*Ivv)))/(2*(Iuu-Ivv))));
        Phiint = real(pi/2 - 1/2*asin(Iuv/(C1int^2-C2int^2)));
        C0int = 8/pi*I0/(C1int*C2int);
        % 3d. adjust results in the way that c1 > c2, 0<Phi<pi
        if C2int>C1int
            aux = C2int;
            C2int = C1int;
            C1int = aux;
            Phiint = Phiint + pi/2;
        end
        while Phiint < 0, Phiint = Phiint + pi; end
        while Phiint >= pi, Phiint = Phiint - pi; end
        % 4.  FIT USING OPTIMIZATION
        % 4c. run the minimization
        opt = optimset('MaxFunEvals', pivPar.spGFitMaxIter*2,'MaxIter',pivPar.spGFitMaxIter);
        auxFit = fminsearch(@(x)objectiveFunc(U0,V0,CCmax,x,cc,pivPar),...  % objective function
            [C1int,C2int,Phiint],... % initial guess [Ufit,Vfit,C1fit,C2fit,Phifit,C0fit],...
            opt);
        Ufit = U0;
        Vfit = V0;
        C1fit = auxFit(1);
        C2fit = auxFit(2);
        Phifit = auxFit(3);
        C0fit = CCmax;
        % 4d. modify results in the way that C1 > C2, -pi/2<phi<=pi/2
        if C2fit > C1fit
            aux = C1fit;
            C1fit = C2fit;
            C2fit = aux;
            Phifit = Phifit + pi/2;
        end
        while Phifit<0, Phifit = Phifit+pi; end
        while Phifit>=pi, Phifit = Phifit-pi; end
        % 4e. if out of bounds, set fail flag of status
        if (Ufit<=-pivPar.spDeltaXNeg)||(Ufit>=pivPar.spDeltaXPos)||...
                (Vfit<=-pivPar.spDeltaYNeg)||(Vfit>=pivPar.spDeltaYPos)||...
                (C1fit<0)||...
                (C2fit<0)||...
                (C0fit>3*max(max(cc)))%(C1fit>1.5*(pivPar.spDeltaXNeg+pivPar.spDeltaXNeg+1))||(C2fit>1.5*(pivPar.spDeltaYNeg+pivPar.spDeltaYNeg+1))||
            pivData.spStatus(ky,kx) = bitset(pivData.spStatus(ky,kx),2);
            Ufit = NaN;
            Vfit = NaN;
            C1fit = NaN;
            C2fit = NaN;
            Phifit = NaN;
            C0fit = 0;
        end
        % 5.  SAVE RESULTS
        pivData.spU0(ky,kx) = U0;
        pivData.spV0(ky,kx) = V0;
        pivData.spUint(ky,kx) = Uint;
        pivData.spVint(ky,kx) = Vint;
        pivData.spUfit(ky,kx) = Ufit;
        pivData.spVfit(ky,kx) = Vfit;
        pivData.spC1int(ky,kx) = C1int;
        pivData.spC2int(ky,kx) = C2int;
        pivData.spC1fit(ky,kx) = C1fit;
        pivData.spC2fit(ky,kx) = C2fit;
        pivData.spPhiint(ky,kx) = Phiint;
        pivData.spPhifit(ky,kx) = Phifit;
        pivData.spC0int(ky,kx) = C0int;
        pivData.spC0fit(ky,kx) = C0fit;
        pivData.spCCmax(ky,kx) = CCmax;
        pivData.spOFfit(ky,kx) = objectiveFunc(Ufit,Vfit,C0fit, [C1fit,C2fit,Phifit],cc,pivPar);
        pivData.spOFint(ky,kx) = objectiveFunc(Uint,Vint,C0int, [C1int,C2int,Phiint],cc,pivPar);
        % give echo
        if (round(auxCount/spEchoInt/20)==auxCount/spEchoInt/20)&&(auxCount>0)
            auxC = datenum(clock);
            auxR = ((nx-kx-1)*ny+ny-ky)/20/spEchoInt*toc(t1);
            auxC = auxC + auxR/24/3600;
            fprintf(' Average time %.4f ms/vector. (Remaining time %.1f min, treatment should finish at %s.)\n',toc(t2)/spEchoInt*1000,auxR/60,datestr(auxC));
            t1 = tic;
        elseif (round(auxCount/spEchoInt) == auxCount/spEchoInt)&&(auxCount>0)
            fprintf(' Average time %.4f ms/vector.\n',toc(t2)/spEchoInt*1000);
        end
        auxCount = auxCount + 1;
    end
    % save partial results
    if pivPar.spOnDrive && ( (round(kx/(10*ceil(spEchoInt/(ny+1))))==kx/(10*ceil(spEchoInt/(ny+1)))) || kx==nx)
        spU0 = pivData.spU0;     %#ok<NASGU>
        spV0 = pivData.spV0;     %#ok<NASGU>
        spUint = pivData.spUint;     %#ok<NASGU>
        spVint = pivData.spVint;     %#ok<NASGU>
        spUfit = pivData.spUfit;     %#ok<NASGU>
        spVfit = pivData.spVfit;     %#ok<NASGU>
        spC1int = pivData.spC1int;     %#ok<NASGU>
        spC2int = pivData.spC2int;     %#ok<NASGU>
        spC1fit = pivData.spC1fit;     %#ok<NASGU>
        spC2fit = pivData.spC2fit;     %#ok<NASGU>
        spPhiint = pivData.spPhiint;   %#ok<NASGU>
        spPhifit = pivData.spPhifit;   %#ok<NASGU>
        spC0int = pivData.spC0int;     %#ok<NASGU>
        spC0fit = pivData.spC0fit;     %#ok<NASGU>
        spCCmax = pivData.spCCmax;     %#ok<NASGU>
        spStatus = pivData.spStatus;   %#ok<NASGU>
        spOFfit = pivData.spOFfit;        %#ok<NASGU>
        spOFint = pivData.spOFint;        %#ok<NASGU>
        auxNewFile = [prefix,fname1,'_',fname2,'_',fname3,'_',fname4,'_',num2str(kx),'.mat'];
        fprintf('    Writing partial results to %s...',auxNewFile);
        tic;
        save([pivPar.spTargetPath,filesep,auxNewFile],...
            'spU0','spV0','spUint','spVint','spUfit','spVfit',...
            'spC0int','spC0fit','spC1int','spC2int','spC1fit','spC2fit',...
            'spPhiint','spPhifit',...
            'spCCmax','spOFint','spOFfit','spStatus',...
            '-v6');
        if isfield(pivPar,'spLockFile') && numel(pivPar.spLockFile)>0
            flock = fopen(pivPar.spLockFile,'w');
            fprintf(flock,[datestr(clock) '\nWriting ' auxNewFile]);
            fclose(flock);
        end
        if numel(LastFile)>0,
            delete([pivPar.spTargetPath,filesep,LastFile]);
        end
        LastFile = auxNewFile;
        fprintf(' Finished in %.2fs.\n', toc);
    end
end
end



function [ObjF,fit] = objectiveFunc(U,V,CCmax,x,cc,pivPar) 
% vector x: [x0,y0,C1,C2,phi,CCmax]'
% do easy names
C1 = x(1);
C2 = x(2);
phi = x(3);
% create variables with coordinates
X = -pivPar.spDeltaXNeg:pivPar.spDeltaXPos;
Y = -pivPar.spDeltaYNeg:pivPar.spDeltaYPos;
[X,Y] = meshgrid(X,Y);
% fitting function
fit = CCmax * exp(-8/C1.^2*((X-U)*cos(phi)-(Y-V)*sin(phi)).^2-8/C2.^2*((X-U)*sin(phi)+(Y-V)*cos(phi)).^2);
% compute difference squared. 
diffsq = (fit-cc).^2;
ObjF = sum(sum(diffsq));
end

