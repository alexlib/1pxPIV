% Script to run example 7 with the available data
cd('PIVsuite v.0.8.3');

clear;
fprintf('\nRUNNING EXAMPLE_07 with available data...\n');
tAll = tic;

%% Set processing parameters

% Use the Test Tububu directory which we know exists
imagePath = ['..',filesep,'Data',filesep,'Test Tububu'];

% define path where processing results will be stored
pivPar.spTargetPath = ['..' filesep 'Data' filesep 'Test Tububu - spOut'];

% Create the output directory if it doesn't exist
if ~exist(pivPar.spTargetPath, 'dir')
    mkdir(pivPar.spTargetPath);
end

% Define image mask (we'll skip this since we don't have a mask file)
% pivPar.imMask1 = '../Data Tububu/PlaneA05/Mask.bmp';
% pivPar.imMask2 = '../Data Tububu/PlaneA05/Mask.bmp';

% define the maximum displacements, for which the cross-correlation is evaluated
pivPar.spDeltaXNeg = 4;
pivPar.spDeltaXPos = 4;
pivPar.spDeltaYNeg = 4;
pivPar.spDeltaYPos = 15;
% define binding (decreases resolution, but averages cross-correlation function from neighbouring pixels,
% improving accuracy)
pivPar.spBindX = 2;
pivPar.spBindY = 4;

% set remaining parameters to defaults
pivPar = pivParams([],pivPar,'defaults1Px');


%% Analyze images

% get list of images
aux = dir([imagePath, filesep, '*.bmp']);
for kk = 1:numel(aux)
    fileList{kk} = [imagePath, filesep, aux(kk).name];  %#ok<SAGROW>
end
fileList = sort(fileList);

% Create image sequences
[im1,im2] = pivCreateImageSequence(fileList,pivPar);

% do PIV analysis
pivData1Px = [];
pivData1Px = pivSinglepixAnalyze(im1,im2,pivData1Px,pivPar);

% Save results
save([pivPar.spTargetPath filesep 'pivData.mat'],'pivData1Px');

%% Show results
% Plot the velocity field
figure(1);
pivQuiver(pivData1Px,...
    'spV0','clipLo',-5,'clipHi',5,...
    'quiver','linespec','-k');
title('Single-pixel PIV analysis results');

fprintf('EXAMPLE_07... FINISHED in %.1f min.\n\n',toc(tAll)/60);
