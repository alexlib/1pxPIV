% Example 01 - simple use of PIVsuite for obtaining the velocity field from a pair of images

clear;
%% 1. Define image pair and image mask
im1 = ['..',filesep,'Data',filesep,'Test von Karman',filesep,'PIVlab_Karman_01.bmp'];
im2 = ['..',filesep,'Data',filesep,'Test von Karman',filesep,'PIVlab_Karman_02.bmp'];
imMask = ['..',filesep,'Data',filesep,'Test von Karman',filesep,'PIVlab_Karman_mask.png'];


%% 2. Define evaluation parameters

% initialize parameters and results
pivPar = [];      % initialize treatment pivParameters
pivData = [];     % initialize detailed results that will go to this structure

% set the masking image (this step can be skiped if no masking is required)
pivPar.imMask1 = imMask;
pivPar.imMask2 = imMask;

% set other parameters to defaults
[pivPar, pivData] = pivParams(pivData,pivPar,'defaults');

% Hint: examine content of structure pivPar to see what are default settings for PIV analysis (type "pivPar"
% to Matlab command line)


%% 3. Run the analysis
[pivData] = pivAnalyzeImagePair(im1,im2,pivData,pivPar);

% give information about computational time and about how much vectors are invalid
fprintf('Elapsed time %.2f s (last pass %.2f s), subpixel interpolation failed for %.2f%% vectors, %.2f%% of vectors rejected.\n', ...
    sum(pivData.infCompTime), pivData.infCompTime(end), ...
    pivData.ccSubpxFailedN/pivData.N*100, pivData.spuriousN/pivData.N*100);

% Hint: examine content of structure pivData to see what is the structure of results of PIV analysis (type 
% "pivData" to Matlab command line)


%% 4. Show results

figure(2);
pivQuiver(pivData,...
    'ccPeak',...                                        % show coherence level
    'quiver','selectStat','replaced','linespec','-w');  % show quiver with replaced vectors shown by white 

figure(1);
pivQuiver(pivData,...
    'Umag',...                                          % show background with magnitude 
    'quiver','selectStat','valid','linespec','-k',...   % show quiver with valid vectors shown by black 
    'quiver','selectStat','replaced','linespec','-w');  % show quiver with replaced vectors shown by white 

