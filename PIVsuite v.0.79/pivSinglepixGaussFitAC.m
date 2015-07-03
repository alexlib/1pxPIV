function [pivData] = pivSinglepixGaussFitAC(pivDataIn,pivPar)

fprintf('Fitting auto-correlation peaks by 2D gaussian function...\n');

% initialize fields and variables

pivData = pivDataIn;
nx = size(pivData.spCC,2);
ny = size(pivData.spCC,1);
aux = zeros(ny,nx) + NaN;
if ~isfield(pivData,'spACC0int'), pivData.spACC0int = aux; end
if ~isfield(pivData,'spACC1int'), pivData.spACC1int = aux; end
if ~isfield(pivData,'spACC2int'), pivData.spACC2int = aux; end
if ~isfield(pivData,'spACC0fit'), pivData.spACC0fit = aux; end
if ~isfield(pivData,'spACC1fit'), pivData.spACC1fit = aux; end
if ~isfield(pivData,'spACC2fit'), pivData.spACC2fit = aux; end
if ~isfield(pivData,'spACPhiint'), pivData.spACPhiint = aux; end
if ~isfield(pivData,'spACPhifit'), pivData.spACPhifit = aux; end
if ~isfield(pivData,'spACmax'), pivData.spACmax = aux; end
if ~isfield(pivData,'spACOFfit'), pivData.spACOFfit = aux; end
if ~isfield(pivData,'spACOFint'), pivData.spACOFint = aux; end
if isfield(pivPar,'spEchoInt')
    spEchoInt = pivPar.spEchoInt;
else
    spEchoInt = 40;
end


% check the existance of a file with intermediate results
prefix = ['spAC_GaussFit_' num2str(pivPar.spBindX,'%d') 'x' num2str(pivPar.spBindY,'%d')];
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
        fprintf('    Reading results of AC peak fit from %s...', LastFile);
        tic;
        aux = load([pivPar.spTargetPath,filesep,LastFile]);
        pivData.spACC0int = aux.spACC0int;
        pivData.spACC1int = aux.spACC1int;
        pivData.spACC2int = aux.spACC2int;
        pivData.spACC0fit = aux.spACC0fit;
        pivData.spACC1fit = aux.spACC1fit;
        pivData.spACC2fit = aux.spACC2fit;
        pivData.spACPhiint = aux.spACPhiint;
        pivData.spACPhifit = aux.spACPhifit;
        pivData.spACmax = aux.spACmax;
        pivData.spACOFfit = aux.spACOFfit;
        pivData.spACOFint = aux.spACOFint;
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
            fprintf('    Fitting auto-correlation peak for velocity vector %d of %d...',auxCount,nx*ny);
            t2 = tic;
        end
        cc = squeeze(pivDataIn.spAC(ky,kx,:,:));
        % 2.  SUBPIXEL INTERPOLATION TO FIND MAXIMA (similar to standard PIV)
        % 2a. obtain initial guess for the CC peak
        [~,Upx] = max(max(cc));    % guess based on maxima of CC
        [CCmax,Vpx] = max(cc(:,Upx));
        % 2b. if maxima is on border of CC, do not interpolate...
        if Upx <= 1 || Vpx <= 1 || Upx >= size(cc,2)-1 || Vpx >= size(cc,1)-1
            Upx = Upx - pivPar.spDeltaAutoCorr - 1;
            Vpx = Vpx - pivPar.spDeltaAutoCorr - 1;
            U0 = Upx;
            V0 = Vpx;         
        else
        % 2c. find initial guess of (U0,V0) based on 2x3 point gaussian fit
            y11 = log(cc(Vpx,Upx-1));
            y12 = log(cc(Vpx,Upx));
            y13 = log(cc(Vpx,Upx+1));
            y21 = log(cc(Vpx-1,Upx));
            y22 = log(cc(Vpx,Upx));
            y23 = log(cc(Vpx+1,Upx));
            U0 = (y11 - y13)/(y11+y13-2*y12)/2;
            V0 = (y21 - y23)/(y21+y23-2*y22)/2;
            Upx = Upx - pivPar.spDeltaAutoCorr - 1;
            Vpx = Vpx - pivPar.spDeltaAutoCorr - 1;
            U0 = Upx + U0;
            V0 = Vpx + V0;
            CCmax = exp(y12 -(y11-y13)^2/(y11+y13-2*y12)/8 -(y21-y23)^2/(y21+y23-2*y22)/8);
        end
        % 2d. if fails, use only maximum with integer-pixel accuracy
        if ~isreal(U0) ||~isreal(V0)
            U0 = Upx;
            V0 = Vpx;
        end
        % 3.  COMPUTATION OF PEAK PARAMETERS USING INTEGRAL VALUES OF CC
        % 3a. create matrices with coordinates
        X = ones(size(cc,1),1) * (-pivPar.spDeltaAutoCorr:pivPar.spDeltaAutoCorr);
        Y = (-pivPar.spDeltaAutoCorr:pivPar.spDeltaAutoCorr)' * ones(1,size(cc,2));
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
        pivData.spACC1int(ky,kx) = C1int;
        pivData.spACC2int(ky,kx) = C2int;
        pivData.spACC1fit(ky,kx) = C1fit;
        pivData.spACC2fit(ky,kx) = C2fit;
        pivData.spACPhiint(ky,kx) = Phiint;
        pivData.spACPhifit(ky,kx) = Phifit;
        pivData.spACC0int(ky,kx) = C0int;
        pivData.spACC0fit(ky,kx) = C0fit;
        pivData.spACmax(ky,kx) = CCmax;
        pivData.spACOFfit(ky,kx) = objectiveFunc(Ufit,Vfit,C0fit, [C1fit,C2fit,Phifit],cc,pivPar);
        pivData.spACOFint(ky,kx) = objectiveFunc(Uint,Vint,C0int, [C1int,C2int,Phiint],cc,pivPar);
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
        spACC1int = pivData.spACC1int;     %#ok<NASGU>
        spACC2int = pivData.spACC2int;     %#ok<NASGU>
        spACC1fit = pivData.spACC1fit;     %#ok<NASGU>
        spACC2fit = pivData.spACC2fit;     %#ok<NASGU>
        spACPhiint = pivData.spACPhiint;   %#ok<NASGU>
        spACPhifit = pivData.spACPhifit;   %#ok<NASGU>
        spACC0int = pivData.spACC0int;     %#ok<NASGU>
        spACC0fit = pivData.spACC0fit;     %#ok<NASGU>
        spACmax = pivData.spACmax;     %#ok<NASGU>
        spACOFfit = pivData.spACOFfit;        %#ok<NASGU>
        spACOFint = pivData.spACOFint;        %#ok<NASGU>
        auxNewFile = [prefix,fname1,'_',fname2,'_',fname3,'_',fname4,'_',num2str(kx),'.mat'];
        fprintf('    Writing partial results to %s...',auxNewFile);
        tic;
        save([pivPar.spTargetPath,filesep,auxNewFile],...
            'spACC0int','spACC0fit','spACC1int','spACC2int','spACC1fit','spACC2fit',...
            'spACPhiint','spACPhifit',...
            'spACmax','spACOFint','spACOFfit',...
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
X = -pivPar.spDeltaAutoCorr:pivPar.spDeltaAutoCorr;
Y = -pivPar.spDeltaAutoCorr:pivPar.spDeltaAutoCorr;
[X,Y] = meshgrid(X,Y);
% fitting function
fit = CCmax * exp(-8/C1.^2*((X-U)*cos(phi)-(Y-V)*sin(phi)).^2-8/C2.^2*((X-U)*sin(phi)+(Y-V)*cos(phi)).^2);
% compute difference squared. 
diffsq = (fit-cc).^2;
ObjF = sum(sum(diffsq));
end

