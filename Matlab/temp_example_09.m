% Example 9: Processing a pair of BOS (Background-Oriented Schlieren) images
% This example demonstrates how to process a pair of BOS images using PIVsuite

clear;
fprintf('\nRUNNING EXAMPLE_09_BOS_IMAGE_PAIR...\n');
tAll = tic;

%% Set processing parameters
pivPar = [];

% Define paths to images
im1Path = fullfile('..', 'Data', 'Test BOS', '11-49-28.000-4.tif');
im2Path = fullfile('..', 'Data', 'Test BOS', '11-49-28.000-6.tif');

fprintf('Image paths:\n  %s\n  %s\n', im1Path, im2Path);

% Check if images exist
if ~exist(im1Path, 'file') || ~exist(im2Path, 'file')
    error('BOS image files not found. Please check the paths.');
end

% Set PIV parameters
pivPar.iaSizeX = 32;     % Interrogation area size in X
pivPar.iaSizeY = 32;     % Interrogation area size in Y
pivPar.iaStepX = 16;     % Interrogation area step in X
pivPar.iaStepY = 16;     % Interrogation area step in Y
pivPar.ccMaxDisplacement = 0.7;  % Maximum displacement allowed (as a fraction of the interrogation area size)
pivPar.smMethod = 'none'; % Smoothing method
pivPar.imMask1 = '';     % No mask for the first image
pivPar.imMask2 = '';     % No mask for the second image
pivPar.imDeform = 'off'; % Image deformation
pivPar.iaMethod = 'defspline'; % Interrogation area method
pivPar.iaImageInterpolationMethod = 'spline'; % Image interpolation method
pivPar.iaSubpixelEstimationMethod = 'gaussianfit'; % Subpixel estimation method

% Set remaining parameters to defaults
pivPar = pivParams([],pivPar,'defaults');

%% Load and preprocess images
fprintf('Loading images...\n');
im1 = imread(im1Path);
im2 = imread(im2Path);

% Convert to double if needed
if ~isa(im1, 'double')
    im1 = double(im1);
end
if ~isa(im2, 'double')
    im2 = double(im2);
end

% If images are RGB, convert to grayscale
if size(im1, 3) > 1
    im1 = rgb2gray(im1);
end
if size(im2, 3) > 1
    im2 = rgb2gray(im2);
end

fprintf('Images loaded. Size: %d x %d pixels\n', size(im1, 2), size(im1, 1));

%% Perform PIV analysis
fprintf('Performing PIV analysis...\n');
pivData = pivAnalyzeImagePair(im1, im2, pivPar);

%% Display results
% Plot the velocity field
figure(1);
pivQuiver(pivData);
title('BOS Image Pair - Velocity Field');

% Plot the velocity magnitude
figure(2);
pivPlot(pivData);
title('BOS Image Pair - Velocity Magnitude');

% Plot the velocity components
figure(3);
subplot(1,2,1);
pivPlot(pivData, 'type', 'u');
title('BOS Image Pair - X Velocity Component');

subplot(1,2,2);
pivPlot(pivData, 'type', 'v');
title('BOS Image Pair - Y Velocity Component');

% Save the results to a .mat file for later analysis
resultFile = 'bos_example_results.mat';
fprintf('Saving results to %s...\n', resultFile);
save(resultFile, 'pivData', 'im1', 'im2', 'pivPar');


% Save the results to a .mat file for later analysis
resultFile = 'bos_example_results.mat';
fprintf('Saving results to %s...\n', resultFile);
save(resultFile, 'pivData', 'im1', 'im2', 'pivPar');

fprintf('EXAMPLE_09_BOS_IMAGE_PAIR... FINISHED in %.1f sec.\n\n', toc(tAll));
