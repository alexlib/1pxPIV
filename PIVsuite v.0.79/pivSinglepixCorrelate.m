function [pivData1Px] = pivSinglepixCorrelate(im1,im2,pivDataIn,pivParIn,corrtype)
% pivSinglepixCorrelate - computes ensemble auto- or cross-correlation for a set of images
%
% Usage:
%   1. [pivData] = pivSinglepixCorrelate(im1,im2,pivData,pivPar)
%        or  [pivData] = pivSinglepixCorrelate(im1,im2,pivData,pivPar,'cross')
%   2. [pivData] = pivSinglepixCorrelate(im1,im2,pivData,pivPar,'auto')
% following usages are also accepted, but not recommended (it is used as self-call from pivSinglepixCorrelate)
%   3. [pivData] = pivSinglepixCorrelate(im1,im2,pivData,pivPar,'auto1')
%   4. [pivData] = pivSinglepixCorrelate(im1,im2,pivData,pivPar,'auto2')
%   5. [pivData] = pivSinglepixCorrelate(im1,im2,pivData,pivPar,'ccauto1')
%   6. [pivData] = pivSinglepixCorrelate(im1,im3,pivData,pivPar,'ccauto2')
%
% Usage 1: Calculates cross-correlation function for image pairs defined by lists im1 and im2.
%
% Usage 2: Calculates auto-correlation function for either images listed in im1, or in im2, or inboth lists, depending
% on pivPar.spACsource.
%
% Usages 3-6 (not recommended for external calls): Computes autocorreletion function of either im1 or im2. Used for
% selfcalls from usage 2 in order to reuse the code for cross-correlation (Usage 1) to compute autocorrelations. 
%
% Inputs:
%    im1, im2 ... lists containing paths to first and second image in each pair
%    pivData ... (struct) structure containing more detailed results. No fields are required. Any field existing in this
%        structure is copied to the output structure pivData.
%    pivPar ... (struct) parameters defining the evaluation. Following fields are considered:
%        spDeltaXNeq, spDeltaXPos ... defines maximum displacement in X direction, for which cross-correlation function 
%            is evaluated
%        spDeltaYNeq, spDeltaYPos ... defines maximum displacement in Y direction, for which cross-
%            correlation function is evaluated
%        spDeltaAutoCorr ... maximum displacement in both X and Y directions, for which auto-correlation 
%            function is evaluated (usage 2)
%        spBindX, spBindY ... defines binding of image pixels. For example, if spBindX=2 and spBindY=4, 
%            cross-correlations of 2x4 neighbouring pixels are averaged before velocity field is computed. 
%            Binding decreases resulting resolution.
%        spACsource ... defines, which image is used for computing the autocorrelation function (usage 2).
%            (pivSinglepixCorrelate.m). Possible values are 
%            'both' ... the autocorralation is evaluated for both first and second image in each pair and 
%                the average of these two autocorrelations is taken in account; use this option when first 
%                and second laser pulses are not the same
%            'im1', 'im2' ... only first or second frame in each pair is used for evaluation of the
%                auto-correlation function.
%        spOnDrive ... determines, whether the processing results are stored on disk.
%        spForceProcessing ... determines, whether the processing should be performed, even if an output 
%            file with intermediate result exists
%        spSaveInterval ... defines, how many images are correlated before storing an intermediate result
%            (pivSinglepixCorrelate.m)
%
% Outputs:
%    pivData  ... (struct) structure containing more detailed results. Following fields are added or updated:
%        spX, spY ... positions, in which cross-correlation function is computed
%        spStatus ... matrix with statu of velocity vectors (uint8). Bits have this coding:
%            1 ... masked (set by pivSinglepixCorrelate)
%            2 ... not used
%            4 ... peak detection failed (set by xxx)
%            8 ... indicated as spurious by median test (set by xxx)
%           16 ... interpolated (set by xxx)
%           32 ... smoothed (set by xxx)
%        spCC, spAC  ... table with values of cross-correlation and autocorrelation peak. This is a 4D array with this 
%            order of dimensions: i. vertical pixel position, ii.  horizontal pixel position, iii. vertical displacement,
%            iv.  horizontal displacement. 
%       
%
% Important variables:   
%    ImgAAvg, ImgBAvg ... average image of all images
%    ImgARms, ImgBRms ... Root-mean square images of all images
%    ccFunc ... cross-correlation function (4D array)
%    validCount ... counts number of pairs appearing in cross-correlation sum for each pixel. The effective
%        number of pairs appearing in the cross-correlation sum is affected by masking; validCount is used to
%        decide whether the corresponding pixel is masked or not.
%
%
% This subroutine is a part of
%
% =========================================
%               PIVsuite
% =========================================
%
% PIVsuite is a set of subroutines intended for processing of data acquired with PIV (particle image
% velocimetry) within Matlab environment. Images can be evaluated either using window-correlation (standard) 
% or single-pixel correlation techniques.
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
%    The standard (window-correlation) algorithm of PIVsuite is a redesigned and enlarged version of PIVlab 
%    software [3], developped by W. Thielicke and E. J. Stamhuis. Some parts of this code are copied or 
%    adapted from it (especially from its piv_FFTmulti.m subroutine). 
%    PIVsuite uses 3rd party software:
%        inpaint_nans.m, by J. D'Errico, [2]
%        smoothn.m, by Damien Garcia, [5]
%    The single-pixel correlation algorithm is implemented following Ref. [6].
%        
% References:
%   [1] Adrian & Whesterweel, Particle Image Velocimetry, Cambridge University Press 2011
%   [2] John D'Errico, inpaint_nans subroutine, http://www.mathworks.com/matlabcentral/fileexchange/4551
%   [3] W. Thielicke and E. J. Stamhuis, PIVlab 1.31, http://pivlab.blogspot.com
%   [4] Raffel, Willert, Wereley & Kompenhans, Particle Image Velocimetry: A Practical Guide. 2nd edition,
%       Springer 2007
%   [5] Damien Garcia, smoothn subroutine, http://www.mathworks.com/matlabcentral/fileexchange/274-smooth
%   [6] S. Scharnowski, R. Hain and C. J. Kahler: Reynolds stress estimation up to single-pixel resolution using 
%       PIV-measurements. Experiments in Fluids 52(2012): 985-1002, http://dx.doi.org/10.1007/s00348-011-1184-1 .






%% 0. Preparations

pivData1Px = pivDataIn;

% create output folder, if needed
if pivParIn.spOnDrive
    if ~exist(pivParIn.spTargetPath,'dir')
        mkdir(pivParIn.spTargetPath);
    end
end

% read first image, get its size
img1 = imread(im1{1});
[sizeY,sizeX] = size(img1);
[X,Y] = meshgrid(1:sizeX,1:sizeY);

% if corrtype is not specified, compute cross-correlation
if nargin <5, corrtype = 'cross'; end

% switch behavior in dependence if cross-correlation, or autocorrelation is required
switch lower(corrtype)
    case 'auto'  % autocorrelation is required
        switch lower(pivParIn.spACsource)
            case 'both'
                pivData1 = pivSinglepixCorrelate(im1,im2,pivDataIn,pivParIn,'auto1');
                pivData2 = pivSinglepixCorrelate(im1,im2,pivDataIn,pivParIn,'auto2');
                pivData1Px.spAC = (pivData1.spAC+pivData2.spAC)./2;
            case 'im1'
                pivData1 = pivSinglepixCorrelate(im1,im2,pivDataIn,pivParIn,'auto1');
                pivData1Px.spAC = pivData1.spAC;
            case 'im2'
                pivData2 = pivSinglepixCorrelate(im1,im2,pivDataIn,pivParIn,'auto2');
                pivData1Px.spAC = pivData2.spAC;
        end
    case 'auto1'  % this case is run only by a self-call from corrtype 'auto'
        pivData1Px = pivDataIn;
        pivPar1 = pivParIn;
        pivPar1.imMask2 = pivParIn.imMask1;
        pivPar1.spDeltaXNeg = pivParIn.spDeltaAutoCorr;
        pivPar1.spDeltaXPos = pivParIn.spDeltaAutoCorr;
        pivPar1.spDeltaYNeg = pivParIn.spDeltaAutoCorr;
        pivPar1.spDeltaYPos = pivParIn.spDeltaAutoCorr;
        pivData1 = pivSinglepixCorrelate(im1,im1,pivDataIn,pivPar1,'ccauto1');
        pivData1Px.spAC = pivData1.spCC;
    case 'auto2'  % this case is run only by a self-call from corrtype 'auto'
        pivData1Px = pivDataIn;
        pivPar2 = pivParIn;
        pivPar2.imMask1 = pivParIn.imMask2;
        pivPar2.spDeltaXNeg = pivParIn.spDeltaAutoCorr;
        pivPar2.spDeltaXPos = pivParIn.spDeltaAutoCorr;
        pivPar2.spDeltaYNeg = pivParIn.spDeltaAutoCorr;
        pivPar2.spDeltaYPos = pivParIn.spDeltaAutoCorr;
        pivData2 = pivSinglepixCorrelate(im2,im2,pivDataIn,pivPar2,'ccauto2');
        pivData1Px.spAC = pivData2.spCC;
        
        % CASE 'CROSS': algorithm for computing cross-correlation follows. This algorithm is used also for auto-correlations.
    case {'cross','ccauto1','ccauto2'}
        % either cross-correlation, or self-call for computing autocorrelation using a cross-correlation
        % algorithm ('corr' and 'ccautoX' differ only in different echo messages and saved/loaded files
        
%% 1 here is the computations of CC        
        % make easy names
        dXNeg = pivParIn.spDeltaXNeg;
        dXPos = pivParIn.spDeltaXPos;
        dYNeg = pivParIn.spDeltaYNeg;
        dYPos = pivParIn.spDeltaYPos;
        spBindX = pivParIn.spBindX;
        spBindY = pivParIn.spBindY;
        spStepX = pivParIn.spStepX;
        spStepY = pivParIn.spStepY;
        
        % echo and prefix for filenames
        switch lower(corrtype)
            case 'cross'
                fprintf('Evaluating single-pixel cross-correlation function...\n');
                prefix = 'spCC_';
            case 'ccauto1'
                fprintf('Evaluating single-pixel auto-correlation function (image 1)...\n');
                prefix = 'spAC1_';
            case 'ccauto2'
                fprintf('Evaluating single-pixel auto-correlation function (image 2)...\n');
                prefix = 'spAC2_';
        end
        tAll = tic;
        
        % get filenames used in filenames for result files
        [~, filename1] = treatImgPath(im1{1});
        [~, filename2] = treatImgPath(im2{1});
        [~, filename3] = treatImgPath(im1{2});
        [~, filename4] = treatImgPath(im2{end});
        
        % read image mask, if there is single mask for all images
        imMask1 = ones(size(img1));
        if isfield(pivParIn,'imMask1')
            if ischar(pivParIn.imMask1)       % single mask for all images, specified by filename
                pivData1Px.imMaskFilename1 = pivParIn.imMask1;
                if ~isempty(pivParIn.imMask1), imMask1 = double(imread(pivParIn.imMask1)); end
            elseif isnumeric(pivParIn.imMask1) && numel(pivParIn.imMask1)==0 % empty mask = no masking
                pivData1Px.imMaskFilename1 = '';
                imMask1 = ones(size(img1));
            elseif isnumeric(pivParIn.imMask1) && numel(pivParIn.imMask1)>0 % single mask for all images, specified by mask
                pivData1Px.imMaskFilename1 = '?';
                imMask1 = double(pivParIn.imMask1);
            elseif iscell(pivParIn.imMask1)   % each image has its own mask
                pivData1Px.imMaskFilename1 = pivParIn.imMask1;
            end
        end
        imMask2 = ones(size(img1));
        if isfield(pivParIn,'imMask2')
            if ischar(pivParIn.imMask2)      % single mask for all images, specified by filename
                pivData1Px.imMaskFilename2 = pivParIn.imMask2;
                if ~isempty(pivParIn.imMask2), imMask2 = double(imread(pivParIn.imMask2)); end
            elseif isnumeric(pivParIn.imMask2) && numel(pivParIn.imMask2)==0 % empty mask = no masking
                pivData1Px.imMaskFilename2 = '';
                imMask2 = ones(size(img1));
            elseif isnumeric(pivParIn.imMask2) && numel(pivParIn.imMask2)>0 % single mask for all images, specified by mask
                pivData1Px.imMaskFilename2 = '?';
                imMask2 = double(pivParIn.imMask2);
            elseif iscell(pivParIn.imMask2)   % each image has its own mask
                pivData1Px.imMaskFilename2 = pivParIn.imMask2;
            end
        end
        
%% 2 Average image        
% compute average and rms images (use identity rms = sqrt(E(x^2)-(E(x))^2)  ). Masked pixels are not taken in account.
        
        % check if file with results already exists
        auxMeanComputed = false;
        if pivParIn.spOnDrive && ~pivParIn.spForceProcessing
            if exist([pivParIn.spTargetPath filesep prefix 'AvgImgs_' ...
                    filename1 '_' filename2 '_' filename3 '_' filename4 '.mat'],'file')
                t1 = tic;
                fprintf('    Reading average and rms images from file %s...',...
                    [prefix 'AvgImgs_' filename1 '_' filename2 '_' filename3 '_' filename4 '.mat']);
                auxMeanComputed = true;
                aux = load([pivParIn.spTargetPath filesep ...
                    prefix 'AvgImgs_' filename1 '_' filename2 '_' filename3 '_' filename4 '.mat']);
                ImgAAvg = aux.ImgAAvg;
                ImgARms = aux.ImgARms;
                ImgBAvg = aux.ImgAAvg;
                ImgBRms = aux.ImgARms;
                fprintf(' Finished in %.2fs.\n',toc(t1));
            end
        end
        
        % If results were not read, compute average and rms images
        if ~auxMeanComputed
            fprintf('    Computing average and rms images... \n        Image pair no.');
            t1 = tic;
            t2 = tic;
            % variables sum and sum squared of individual pixels
            ImgBSum = zeros(sizeY,sizeX);
            ImgASum = zeros(sizeY,sizeX);
            ImgBSumSq = zeros(sizeY,sizeX);
            ImgASumSq = zeros(sizeY,sizeX);
            ImgBCount = zeros(sizeY,sizeX);
            ImgACount = zeros(sizeY,sizeX);
            % loop through all images
            for ki = 1:numel(im1)
                if round(ki/10)==ki/10
                    fprintf(' %d',ki);
                end
                if round(ki/100)==ki/100
                    fprintf(' of %d (average time %.3fs/pair)\n        Image pair no.',numel(im1),toc(t2)/100);
                    t2 = tic;
                end
                ImgA = imread(im1{ki});
                ImgB = imread(im2{ki});
                % mask images
                if iscell(pivParIn.imMask1), imMask1 = double(imread(pivParIn.imMask1{ki})); end
                if iscell(pivParIn.imMask2), imMask2 = double(imread(pivParIn.imMask2{ki})); end
                imMask1(imMask1>0) = 1;
                imMask2(imMask2>0) = 1;
                ImgA = double(ImgA).*imMask1;
                ImgB = double(ImgB).*imMask2;
                % compute sum and sum squared
                ImgASum = ImgASum + ImgA;
                ImgASumSq = ImgASumSq + ImgA.^2;
                ImgACount = ImgACount + imMask1;
                ImgBSum = ImgBSum + ImgB;
                ImgBSumSq = ImgBSumSq + ImgB.^2;
                ImgBCount = ImgBCount + imMask2;
            end
            fprintf('\n        Finished in %.2f s.\n',toc(t1));
            % compute Avg and Rms images
            auxAMasked = ImgACount==0;
            ImgAAvg = ImgASum./ImgACount;
            ImgARms = sqrt(ImgASumSq./ImgACount-(ImgASum./ImgACount).^2);
            ImgAAvg(auxAMasked) = mean(mean(ImgAAvg(~auxAMasked)));
            ImgARms(auxAMasked) = mean(mean(ImgARms(~auxAMasked)));
            auxBMasked = ImgBCount==0;
            ImgBAvg = ImgBSum./ImgBCount;
            ImgBRms = sqrt(ImgBSumSq./ImgBCount-(ImgBSum./ImgBCount).^2);
            ImgBAvg(auxBMasked) = mean(mean(ImgBAvg(~auxBMasked)));
            ImgBRms(auxBMasked) = mean(mean(ImgBRms(~auxBMasked)));
            % smooth average and rms images
            ImgAAvg = smoothn(ImgAAvg,pivParIn.spAvgSmooth);
            ImgARms = smoothn(ImgARms,pivParIn.spRmsSmooth);
            ImgAAvg(auxAMasked) = NaN;
            ImgARms(auxAMasked) = NaN;
            ImgBAvg = smoothn(ImgBAvg,pivParIn.spAvgSmooth);
            ImgBRms = smoothn(ImgBRms,pivParIn.spRmsSmooth);
            ImgBAvg(auxBMasked) = NaN;
            ImgBRms(auxBMasked) = NaN;
            tic;
            fprintf('    Writing average and rms images to file %s...\n',...
                [prefix 'AvgImgs_' filename1 '_' filename2 '_' filename3 '_' filename4 '.mat']);
            if pivParIn.spOnDrive
                auxFilename = [prefix 'AvgImgs_' ...
                    filename1 '_' filename2 '_' filename3 '_' filename4 '.mat'];
                save([pivParIn.spTargetPath filesep auxFilename], 'ImgAAvg','ImgARms','ImgBAvg','ImgBRms','-v6');
                if isfield(pivParIn,'spLockFile') && numel(pivParIn.spLockFile)>0
                    flock = fopen(pivParIn.spLockFile,'w');
                    fprintf(flock,[datestr(clock) '\nWriting ' auxFilename]);
                    fclose(flock);
                end
                fprintf('        Finished in %.2fs.\n',toc);
            end
        end
        
%% 2 Read previous results, if available        

        % initialize variables for cross-correlation and autocorrelation function
        ccFunc = zeros(sizeY,sizeX,1+dYNeg+dYPos,1+dXNeg+dXPos);
        validCount = zeros(sizeY,sizeX);
        imgNans = zeros(sizeY+dYNeg+dYPos,sizeX+dXNeg+dXPos)+NaN;

        % check the existence of an output file with cross-correlation. If some file exists, read it and continue
        % cross-correlation only for missing data.
        if pivParIn.spOnDrive && ~pivParIn.spForceProcessing
            % get the name of last file, for which the result is stored
            aux = dir([pivParIn.spTargetPath,filesep,prefix,filename1,'_',filename2,'_',filename3,'*.mat']);
            auxfiles = cell(numel(aux,1));
            for ki = 1:numel(aux)
                auxfiles{ki,1} = aux(ki).name;
            end
            if numel(aux)>0
                auxfiles = sort(auxfiles);
                LastFile = auxfiles{end};
                auxCharsToRemove = numel([prefix,filename1,'_',filename2,'_',filename3])+2;
                auxLastIm = LastFile(auxCharsToRemove:end-4);
                % find the last treated file in the list of images
                aux = strfind(im2,auxLastIm);
                PairToStart = 1;
                for kk=1:numel(aux)
                    if numel(aux{kk})>0, PairToStart=kk+1; end
                end
            else
                PairToStart = 1;
                LastFile = '';
            end
        else
            PairToStart = 1;
            LastFile = '';
        end
        if PairToStart > 1
            fprintf('    Reading partial results from %s...', LastFile);
            tic;
            aux = load([pivParIn.spTargetPath filesep LastFile]);
            ccFunc = aux.ccFunc;
            validCount = aux.validCount;
            fprintf(' Finished in %.2fs.\n', toc);
        end
        
%% 3 Cross-correlate image pairs        
        % Cross-correlate image pairs
        t1 = tic;
        for ki = PairToStart:numel(im1)
            [~,auxName1,auxExt1] = fileparts(im1{ki});
            [~,auxName2,auxExt2] = fileparts(im2{ki});
            if isfield(pivParIn,'expName')
                auxstr = pivParIn.expName;
            else
                auxstr = '???';
            end
            switch lower(corrtype)
                case 'cross'
                    fprintf('    Cross-correlating image pair no. %d of %d (%s: %s%s, %s%s)...',ki,numel(im1),auxstr,auxName1,auxExt1,auxName2,auxExt2);
                case 'ccauto1'
                    fprintf('    Auto-correlating image1 no. %d of %d (%s: %s%s)...',ki,numel(im1),auxstr,auxName1,auxExt1);
                case 'ccauto2'
                    fprintf('    Auto-correlating image2 no. %d of %d (%s: %s%s)...',ki,numel(im1),auxstr,auxName1,auxExt1);
            end
            t2 = tic;
            % read mask, if different for each frame
            if iscell(pivParIn.imMask1), imMask1 = double(imread(pivParIn.imMask1{ki})); end
            if iscell(pivParIn.imMask2), imMask2 = double(imread(pivParIn.imMask2{ki})); end
            imMask1(imMask1>0) = 1;
            imMask2(imMask2>0) = 1;
            validCount = validCount + imMask1.*imMask2;
            % read images, normalize them and mask them
            ImgA = double(imread(im1{ki}));
            ImgB = double(imread(im2{ki}));
            ImgA = (ImgA-ImgAAvg)./ImgARms;
            ImgB = (ImgB-ImgBAvg)./ImgBRms;
            ImgA(imMask1==0) = NaN;
            ImgB(imMask2==0) = NaN;
            % pad image 1 (for cross-correlation)
            ImgBPadded = imgNans;
            ImgBPadded(dYNeg+1:dYNeg+sizeY,dXNeg+1:dXNeg+sizeX) = ImgB;
            % loop over all possible paddings of image 2 (for cross-correlation)
            for kdy = -dYNeg:dYPos
                for kdx = -dXNeg:dXPos
                    % NOT CODED: equivalent of iaImageToDeform
                    % pad image 2 (for cross-correlation)
                    ImgAPadded = imgNans;
                    ImgAPadded(dYNeg+1+kdy:dYNeg+sizeY+kdy,dXNeg+1+kdx:dXNeg+sizeX+kdx) = ImgA;
                    % compute cross-correlation
                    ccPadded = ImgBPadded.*ImgAPadded;
                    % remove padding
                    ccImgPair = ccPadded(dYNeg+1:dYNeg+sizeY,dXNeg+1:dXNeg+sizeX);
                    % replace NaN's by zeros, but remember them
                    masked = ~isnan(ccImgPair);
                    ccImgPair(~masked) = 0;
                    % add to cross-correlation sum
                    ccFunc(:,:,kdy+dYNeg+1,kdx+dXNeg+1) = ccFunc(:,:,kdy+dYNeg+1,kdx+dXNeg+1) + ccImgPair/double(numel(im1));
                end
            end
            % give echo about evaluation time
            if round(ki/10)==ki/10
                auxC = datenum(clock);
                auxR = (numel(im1)-ki+1)*toc(t1)/(ki+1-PairToStart);
                auxC = auxC + auxR/24/3600;
                fprintf(' Finished in %.2fs. (Remaining time %.1f min, treatment should finish at %s.)\n',toc(t2),auxR/60,datestr(auxC));
            else
                fprintf(' Finished in %.2fs.\n',toc(t2));
            end
            % save correlation results
            if pivParIn.spOnDrive && (ki>1) && ...
                    ((round(ki/pivParIn.spSaveInterval)==ki/pivParIn.spSaveInterval)||(ki==numel(im1)))
                [~, filenameL] = treatImgPath(im2{ki});
                auxFile = [prefix,filename1,'_',filename2,'_',filename3,'_',filenameL,'.mat'];
                fprintf('    Saving intermediate results to file %s...',auxFile);
                ts = tic;
                save([pivParIn.spTargetPath,filesep,auxFile],'ccFunc','validCount','-v6');
                if isfield(pivParIn,'spLockFile') && numel(pivParIn.spLockFile)>0
                    flock = fopen(pivParIn.spLockFile,'w');
                    fprintf(flock,[datestr(clock) '\nWriting ' auxFile]);
                    fclose(flock);
                end
                if numel(LastFile)>0, delete([pivParIn.spTargetPath,filesep,LastFile]); end
                LastFile = auxFile;
                fprintf(' Finished in %.2fs.\n',toc(ts));
            end
        end
        
%% 4 Bind neighbouring pixels        
        % average cross-correlation function across neighbouring pixels, if required
        % calculate average cross-correlation over multiple pixels, only if required
        if spBindX > 1 || spBindY > 1
            % check existence of the output file
            auxBindingDone = false;
            if spBindX~=spStepX || spBindX~=spStepX
                auxBindPrefix = ['Bind' num2str(spBindX,'%d') 'x' num2str(spBindY,'%d') 'step' num2str(spStepX,'%d') 'x' num2str(spStepY,'%d') '_'];
            else
                auxBindPrefix = ['Bind' num2str(spBindX,'%d') 'x' num2str(spBindY,'%d') '_'];
            end
            if pivParIn.spOnDrive && ~pivParIn.spForceProcessing
                if exist([pivParIn.spTargetPath filesep prefix auxBindPrefix ...
                        filename1 '_' filename2 '_' filename3 '_' filename4 '.mat'],'file')
                    t1 = tic;
                    fprintf('    Reading space-averaged results from file %s...',...
                        [prefix auxBindPrefix filename1 '_' filename2 '_' filename3 '_' filename4 '.mat']);
                    auxBindingDone = true;
                    aux = load([pivParIn.spTargetPath filesep ...
                        prefix auxBindPrefix  filename1 '_' filename2 '_' filename3 '_' filename4 '.mat']);
                    X = aux.X;
                    Y = aux.Y;
                    validCount = aux.validCount;
                    ccFunc = aux.ccFunc;
                    fprintf(' Finished in %.2fs.\n',toc(t1));
                end
            end
            % not read from file: bind pixels
            if ~auxBindingDone
                fprintf('    Binding results for neighbouring pixels... \n        Treating row');
                t1 = tic;
                % get position of binding areas (similar to interrogation areas)
                sizeXNew = floor((sizeX-spBindX-1)/spStepX+1);
                sizeYNew = floor((sizeY-spBindY-1)/spStepY+1);
                auxLengthX = spStepX * (sizeXNew-1) + spBindX;
                auxLengthY = spStepY * (sizeYNew-1) + spBindY;
                auxFirstX = floor((sizeX - auxLengthX)/2);
                auxFirstY = floor((sizeY - auxLengthY)/2);
                % nitialize arrays
                auxCC = zeros(sizeYNew,sizeXNew,1+dYNeg+dYPos,1+dXNeg+dXPos);
                auxValidCount = zeros(sizeYNew,sizeXNew);
                auxX = zeros(sizeYNew,sizeXNew);
                auxY = zeros(sizeYNew,sizeXNew);
                % loop over all resulting cross-correlations
                for ky = 1:sizeYNew
                    for kx = 1:sizeXNew
                        % loop over all CC functions, which will be averaged
                        for jx = 1:spBindX
                            for jy = 1:spBindY
                                auxCC(ky,kx,:,:) = auxCC(ky,kx,:,:) + 1/(spBindX*spBindY)*...
                                    ccFunc((ky-1)*spStepY+jy+auxFirstY,(kx-1)*spStepX+jx+auxFirstX,:,:);
                                auxX(ky,kx) = auxX(ky,kx) + 1/(spBindX*spBindY)*...
                                    X((ky-1)*spStepY+jy+auxFirstY,(kx-1)*spStepX+jx+auxFirstX);
                                auxY(ky,kx) = auxY(ky,kx) + 1/(spBindX*spBindY)*...
                                    Y((ky-1)*spStepY+jy+auxFirstY,(kx-1)*spStepX+jx+auxFirstX);
                                auxValidCount(ky,kx) = auxValidCount(ky,kx) + 1/(spBindX*spBindY)*...
                                    validCount((ky-1)*spStepY+jy+auxFirstY,(kx-1)*spStepX+jx+auxFirstX);
                            end
                        end
                    end
                    if round(ky/10)==ky/10
                        fprintf(' %d',ky);
                    end
                    if round(ky/100)==ky/100 && sizeYNew-ky>10
                        fprintf(' of %d\n        Treating row',sizeYNew);
                    elseif round(ky/100)==ky/100
                        fprintf(' of %d\n',sizeYNew);
                    end
                end
                ccFunc = auxCC;
                validCount = auxValidCount;
                X = auxX;
                Y = auxY;
                fprintf('        Finished in %.2fs.\n',toc(t1));
                tic;
                % save bind results to file
                if pivParIn.spOnDrive
                    fprintf('    Writing space-averaged results to file %s...\n',...
                        [prefix auxBindPrefix filename1 '_' filename2 '_' filename3 '_' filename4 '.mat']);
                    auxFilename = [prefix auxBindPrefix ...
                        filename1 '_' filename2 '_' filename3 '_' filename4 '.mat'];
                    save([pivParIn.spTargetPath filesep auxFilename],'X','Y','ccFunc','validCount','-v6');
                    if isfield(pivParIn,'spLockFile') && numel(pivParIn.spLockFile)>0
                        flock = fopen(pivParIn.spLockFile,'w');
                        fprintf(flock,[datestr(clock) '\nWriting ' auxFilename]);
                        fclose(flock);
                    end
                    fprintf('        Finished in %.2fs.\n',toc);
                end
            end
        end
        % indicate "masked" status
        status = uint16(zeros(size(X)));
        status(validCount<0.499*numel(im1)) = uint16(1);
        
%% 5 output results        
        % create output variable
        pivData1Px.spX = X;
        pivData1Px.spY = Y;
        pivData1Px.spStatus = status;
        pivData1Px.spCC = ccFunc;
        switch lower(corrtype)
            case 'cross'
                pivData1Px.spImg1First = filename1;
                pivData1Px.spImg2First = filename2;
                pivData1Px.spImg1Second = filename3;
                pivData1Px.spImg2Last = filename4;
                fprintf('    Cross-correlation evaluation finished in %.1f min.\n',toc(tAll)/60);
            case 'ccauto1'
                fprintf('    Auto-correlation evaluation (image 1) finished in %.1f min.\n',toc(tAll)/60);
            case 'ccauto2'
                fprintf('    Auto-correlation evaluation (image 2) finished in %.1f min.\n',toc(tAll)/60);
        end
        pivData1Px.spDeltaXNeg = dXNeg;
        pivData1Px.spDeltaXPos = dXPos;
        pivData1Px.spDeltaYNeg = dYNeg;
        pivData1Px.spDeltaYPos = dYPos;
        pivData1Px.spBindX = spBindX;
        pivData1Px.spBindY = spBindY;
        pivData1Px.spStepX = spStepX;
        pivData1Px.spStepY = spStepY;
end   % switch lower(corrtype)
end   % pivSinglePixCorrelate - end of main function


%% local functions
function [imgNo, filename, folder] = treatImgPath(path)
% separate the path to get the folder, filename, and number if contained in the name
filename = '';
imgNo = [];
folder = '';
if numel(path)>0
    path = path(end:-1:1);
    I = find(path==filesep);
    I = I(1);
    Idot = find(path=='.');
    Idot = Idot(1);
    try
        folder = path(I+1:end);
        folder = folder(end:-1:1);
    catch  %#ok<CTCH>
        folder = '';
        I = length(path)+1;
    end
    try
        filename = path(Idot+1:I-1);
        filename = filename(end:-1:1);
    catch  %#ok<CTCH>
        filename = '';
    end
    try
        aux = regexp(filename,'[0-9]');
        aux = filename(aux);
        imgNo = str2double(aux);
    catch  %#ok<CTCH>
        imgNo = [];
    end
end
end

