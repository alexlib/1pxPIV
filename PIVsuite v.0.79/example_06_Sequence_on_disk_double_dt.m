% Example 06 - Treat sequences of images. Different frames for evaluate low and high speeds.
% For example, image pairs (Img02,Img03), (Img03,Img04), (Img04,Img05) are used for treating region with high
% velocity or regions with considerable out-of-plane loss of particles (low coherence). Image pairs with
% larger time difference, (Img01,Img04), (Img02,Img05), (Img03,Img06) are used for analyzing regions with low 
% velocity, in which the displacement is too small to be precisely evaluated with smaller time step. Both
% results are then combined in a single velocity field.

clear;

%% Set processing parameters

% path to data
dataPath = ['..',filesep,'Data'];                     % path to folder containing data

% experimental conditions
pivData.expScale = 42.1e-3;                             % (mm/px)
pivData.expFps = 10000;                                 % frame rate (Hz)
pivData.expVmean = (23+23)*1e6/60/88/44;                % mean velocity in the cell (mm/s)
pivData.expVmeanPx = pivData.expVmean/pivData.expFps/pivData.expScale;
                                                        % mean velocity in the cell (px/frame)

% Parameters for processing first pair of images
pivParInit.iaSizeX = [64 32 24];                        % interrogation area size
pivParInit = pivParams([],pivParInit,'defaults');       % set all other parameters to defaults
pivParInit.qvOptionsPair = {...                         % define plot shown between iterations
    'Umag',...                                          
    'quiver','selectStat','valid','linespec','-k',...    
    'quiver','selectStat','replaced','linespec','-w'};

% Parameters for processing image sequence
pivPar.iaSizeX = 24;                                    % interrogation area 16 px
pivPar.anOnDrive = true;                                % store results on a drive during processing
pivPar.anForceProcessing = false;                        % whether re-process data, for which a result file exists
pivPar.qvPair = {...                                   % plot showing results during processing
    'Umag','subtractV',pivData.expVmeanPx,'clipLo',0,'clipHi',3,...     % show background (velocity magnitude) and quiver (velocity 
    'quiver','subtractV',pivData.expVmeanPx,'selectStat','valid',...    % field), both with V decreased by Vmean
    'quiver','subtractV',pivData.expVmeanPx,'selectStat','replaced','linespec','-w'};           
pivPar = pivParams([],pivPar,'DefaultsSeq');           % set other parameters to default values

% define other settings for first processing (dt = 1 frame) 
pivPar1 = pivPar;                                       % settings for first processing 
pivPar1.seqPairInterval = 1;                            % interval between subsequent image pairs
pivPar1.seqFirstIm = 3;                                 % first image of the first image pair
pivPar1.seqDiff = 1;                                    % interval between images within one image pair
pivPar1.seqMaxPairs = Inf;                              % how many image pairs should be processed
pivPar1.anVelocityEst = 'previous';                     % use results of previous image pair for velocity estimate

% define other settings for second processing (dt = 3 frames) 
pivPar2 = pivPar;                                       % settings for second processing
pivPar2.seqPairInterval = 1;                            % interval between subsequent image pairs
pivPar2.seqFirstIm = 2;                                 % first image of the first image pair
pivPar2.seqDiff = 3;                                    % interval between images within one image pair
pivPar2.seqMaxPairs = Inf;                              % how many image pairs should be processed
pivPar2.anVelocityEst = 'pivData';                      % use results of processing with (dt = 1 frame) for 
                                                        %     velocity estimate


%% Process experimental run

expName = 'Test Tububu';                                % name of folder with experimental run
imagePath = [dataPath, filesep, expName];               % path to folder containing images
pivOutput = [dataPath, filesep, 'pivOut - ', expName];  % path with output files

% get list of images
aux = dir([imagePath filesep '*.bmp']);
for kk = 1:numel(aux)
    fileList{kk} = [imagePath, filesep, aux(kk).name];  %#ok<SAGROW>
end
fileList = sort(fileList);

% set the output path
pivPar1.anTargetPath = pivOutput;
pivPar2.anTargetPath = pivOutput;

% processing with dt = 1 frame
[im1,im2] = pivCreateImageSequence(fileList,pivPar1);
pivData1 = pivAnalyzeImageSequence(im1,im2,pivData,pivPar1,pivParInit);

% processing with dt = 3 frames
[im1,im2] = pivCreateImageSequence(fileList,pivPar2);
pivData2 = pivManipulateData('multiplyVelocity',pivData1,3);
pivData2 = pivAnalyzeImageSequence(im1,im2,pivData2,pivPar2);

% combine data from both processings
pivData2 = pivManipulateData('multiplyVelocity',pivData2,1/3);
pivDataC = pivManipulateData('combineData',pivData1,pivData2);

% validate, replace and smooth combined velocity field
pivDataC = pivValidate(pivDataC,pivPar);
pivDataC = pivReplace(pivDataC,pivPar);
pivPar.smMethodSeq = 'smoothn';
pivPar.smSigmaSeq = 0.05;
pivDataC = pivSmooth(pivDataC,pivPar);


%% Show results

figure(1);
for kk=1:size(pivDataC.U,3)
    hold off;
    pivQuiver(pivDataC,'timeSlice',kk,...
        'Umag','subtractV',pivData.expVmeanPx,'clipLo',0,'clipHi',3,...
        'quiver','linespec','-m','selectMult',3,'selectStat','valid','subtractV',pivData.expVmeanPx,...
        'quiver','linespec','-k','selectMult',1,'selectStat','valid','subtractV',pivData.expVmeanPx,...
        'quiver','linespec','-w','selectMult',1,'selectStat','replaced','subtractV',pivData.expVmeanPx);
    title('Background: U_{mag}. Quiver: velocity (cyan: timeStep = 3 frames, black: timestep = 1 frame)');
    F(kk) = getframe;                    %#ok<SAGROW>
end

movie(F,3);
