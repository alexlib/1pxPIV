% Example 07 - Treat sequences of images; use single-pixel correlation to compute average velocity field

clear;
fprintf('\nRUNNING EXAMPLE_07...\n');
tAll = tic;

%% Set processing parameters

% this variable defines, from where images should be read
imagePath = ['..',filesep,'Data',filesep,'Test Tububu'];

% define path, where processing results will be stored
pivPar.spTargetPath = ['..' filesep 'Data' filesep 'Test Tububu - spOut'];

% Define image mask (commented out as we don't have mask files)
% pivPar.imMask1 = '../Data/Test Tububu/Mask.bmp';
% pivPar.imMask2 = '../Data/Test Tububu/Mask.bmp';

% define the maximum displacements, for which the cross-correlation is evaluated
pivPar.spDeltaXNeg = 4;
pivPar.spDeltaXPos = 4;
pivPar.spDeltaYNeg = 4;
pivPar.spDeltaYPos = 15;
% define binding (decreases resolution, but averages cross-correlation function from neighbouring pixels,
% improving accuracy)
pivPar.spBindX = 2;
pivPar.spBindY = 4;

% pivPar.seqPairInterval = 300;
% pivPar.seqMaxPairs = 4652;

% set remaining parameters to defaults
pivPar = pivParams([],pivPar,'defaults1Px');


%% Analyze images

% get list of images
aux = dir([imagePath, filesep, '*.bmp']);
fileList = {};
for kk = 1:numel(aux)
    fileList{kk} = [imagePath, filesep, aux(kk).name];  %#ok<SAGROW>
end
fileList = sort(fileList);
[im1,im2] = pivCreateImageSequence(fileList,pivPar);

% do PIV analysis
pivData1Px = [];
pivData1Px = pivSinglepixAnalyze(im1,im2,pivData1Px,pivPar);


%% Show results
%
% pivQuiver(pivData1Px,...
%     'UmagMean','clipLo',0,'clipHi',3,...
%     'quiverMean','linespec','-k');
% title('Background: U_{mag}. Quiver: velocity');

% auxSize = size(pivData1Px.ccFunc);
% sizeY = auxSize(1);
% sizeX = auxSize(2);
% ccExpanded = zeros(sizeY*(pivPar.spDeltaYNeg+pivPar.spDeltaYPos+2),sizeX*(pivPar.spDeltaXNeg+pivPar.spDeltaXPos+2))-0.1;
% acExpanded = zeros(sizeY*(2*pivPar.spDeltaAutoCorr+2),sizeX*(2*pivPar.spDeltaAutoCorr+2))-0.1;
% Xcc = ((1:sizeX*(pivPar.spDeltaXNeg+pivPar.spDeltaXPos+2))-pivPar.spDeltaXNeg-1)*pivPar.spBindX/(pivPar.spDeltaXNeg+pivPar.spDeltaXPos+2);
% Ycc = ((1:sizeY*(pivPar.spDeltaYNeg+pivPar.spDeltaYPos+2))-pivPar.spDeltaYNeg-1)*pivPar.spBindY/(pivPar.spDeltaYNeg+pivPar.spDeltaYPos+2);
% Xac = ((1:sizeX*(2*pivPar.spDeltaAutoCorr+2))-pivPar.spDeltaAutoCorr-1)*pivPar.spBindX/(2*pivPar.spDeltaAutoCorr+2);
% Yac = ((1:sizeY*(2*pivPar.spDeltaAutoCorr+2))-pivPar.spDeltaAutoCorr-1)*pivPar.spBindY/(2*pivPar.spDeltaAutoCorr+2);
%
% fprintf('Flattenning cross-correlation structure for display... \n    Treating row');
% tic;
% for ky=1:sizeY;
%     for kx=1:sizeX
%         ccExpanded(...
%               (ky-1)*(pivPar.spDeltaYNeg+pivPar.spDeltaYPos+2)+1:ky*(pivPar.spDeltaYNeg+pivPar.spDeltaYPos+2)-1,...
%               (kx-1)*(pivPar.spDeltaXNeg+pivPar.spDeltaXPos+2)+1:kx*(pivPar.spDeltaXNeg+pivPar.spDeltaXPos+2)-1)...
%             = squeeze(pivData1Px.ccFunc(ky,kx,:,:));
%         acExpanded(...
%               (ky-1)*(2*pivPar.spDeltaAutoCorr+2)+1:ky*(2*pivPar.spDeltaAutoCorr+2)-1,...
%               (kx-1)*(2*pivPar.spDeltaAutoCorr+2)+1:kx*(2*pivPar.spDeltaAutoCorr+2)-1)...
%             = squeeze(pivData1Px.acFunc(ky,kx,:,:));
%     end;
%     if round(ky/10)==ky/10
%         fprintf(' %d ',ky);
%     end;
%     if round(ky/100)==ky/100 && sizeY-ky>10
%         fprintf(' of %d\n    Treating row',sizeY);
%     elseif round(ky/100)==ky/100
%         fprintf(' of %d\n',sizeY);
%     end;
% end
% fprintf('    Finished in %.2f min.\n',toc/60);
%
% figure(1);
% ccExpanded(ccExpanded<-0.1)=-0.1;
% imagesc(Xcc,Ycc,ccExpanded);
% axis equal;
% colorbar
%
% figure(2);
% acExpanded(acExpanded<-0.1)=-0.1;
% imagesc(Xac,Yac,acExpanded);
% axis equal;
% colorbar

fprintf('EXAMPLE_07... FINISHED in %.1f min.\n\n',toc(tAll)/60);

if ~usejava('desktop')
    exit;
end
