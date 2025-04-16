% Example 05 - Treat sequences of images. Results are stored on disc. Analysis has only single pass. 
% Velocity % estimate is taken from the previous image pair. This type of processing is recommended for 
% treating large number frames, if the record is resolved in time.

clear;

%% Set processing parameters

% this variable defines, from where images should be read 
imagePath = ['..',filesep,'Data',filesep,'Test Tububu'];
pivPar.anTargetPath = ['..',filesep,'Data',filesep,'pivOut - Test Tububu'];


% Parameters for processing first pair of images
pivParInit.iaSizeX = [64 32 24];             % interrogation area size for first, second and third pass, first image pair only
pivPar.iaSizeX = 24;                         % interrogation area size, only single pass, other images
pivPar.anVelocityEst = 'previousSmooth';     % use velocity coming from previous image pair as velocity estimate for image deformation
pivPar.seqPairInterval = 1;                  % process only one image pair on three
pivPar.anOnDrive = true;                     % files with results will be stored here
pivPar.anForceProcessing = true;            % if false, only image pairs, for which no file with results is 
            % available, will be processed. If image pairs should be re-processed, overriding existing output 
            % files, set this to true.
pivPar.anTargetPath = ['..',filesep,'Data',filesep,'pivOut - Test Tububu'];
                                             % directory for storing results
pivPar.qvPair = {...                         % define plot shown between iterations
    'Umag','clipHi',3,...                                          
    'quiver','selectStat','valid','linespec','-k',...    
    'quiver','selectStat','replaced','linespec','-w'};
pivParInit = pivParams([],pivParInit,'defaults');
pivPar = pivParams([],pivPar,'defaultsSeq');


%% Analyze images

totaltime = tic;

% get list of images
aux = dir([imagePath, filesep, '*.bmp']);
for kk = 1:numel(aux)
    fileList{kk} = [imagePath, filesep, aux(kk).name];  %#ok<SAGROW>
end
fileList = sort(fileList);

% do PIV analysis
figure(1);
[im1,im2] = pivCreateImageSequence(fileList,pivPar);
pivData = pivAnalyzeImageSequence(im1,im2,[],pivPar,pivParInit);

fprintf('Analysis finished in %.2f s.\n',toc(totaltime));

%% Show results

figure(1);
for kk=1:size(pivData.U,3)
    hold off;
    pivQuiver(pivData,'timeSlice',kk,...
        'Umag','clipLo',0,'clipHi',3,...
        'quiver','linespec','-k','selectStat','valid',...
        'quiver','linespec','-w','selectStat','replaced');
    title('Background: U_{mag}. Quiver: velocity (black: valid, white: replaced)');
    F(kk) = getframe;                    %#ok<SAGROW>
end

movie(F,3);
