% Example 9: Simple processing of a pair of BOS images
% This example demonstrates how to process a pair of BOS images using PIVsuite

clear;
fprintf('\nRUNNING EXAMPLE_09_BOS_SIMPLE...\n');
tAll = tic;

%% Set processing parameters
pivPar = []; % Initialize empty parameter structure
% Set parameters to defaults
pivPar = pivParams([], pivPar, 'defaults');

% Define paths to images
im1Path = fullfile('..', 'Data', 'Test BOS', '11-49-28.000-4.tif');
im2Path = fullfile('..', 'Data', 'Test BOS', '11-49-28.000-6.tif');

fprintf('Image paths:\n  %s\n  %s\n', im1Path, im2Path);

% Check if images exist
if ~exist(im1Path, 'file') || ~exist(im2Path, 'file')
    error('BOS image files not found. Please check the paths.');
end

% Modify default parameters for faster processing
pivPar.iaSizeX = 64;     % Larger interrogation area for faster processing
pivPar.iaSizeY = 64;
pivPar.iaStepX = 32;     % Larger step size for fewer vectors
pivPar.iaStepY = 32;
pivPar.anNpasses = 1;    % Single pass for speed

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

% Downsample images for faster processing
scale = 0.25; % Process at 25% of original size
im1 = imresize(im1, scale);
im2 = imresize(im2, scale);

fprintf('Images loaded and resized. Size: %d x %d pixels\n', size(im1, 2), size(im1, 1));

%% Perform PIV analysis
fprintf('Performing PIV analysis...\n');
pivData = pivAnalyzeImagePair(im1, im2, pivPar);

%% Display results
% Plot the velocity field
figure(1);
pivQuiver(pivData);
title('BOS Image Pair - Velocity Field');

% Create a quiver plot over the image
fprintf('Creating quiver plot over the image...\n');

% Create a new figure
figure(2);

% Display the image
imagesc(im1);
colormap gray;
hold on;

% Extract the velocity vectors and coordinates
x = pivData.x;
y = pivData.y;
u = pivData.u;
v = pivData.v;

% Plot the quiver
quiver(x, y, u, v, 'r', 'LineWidth', 1.5);

% Add title and adjust the plot
title('BOS Image with Velocity Field', 'FontSize', 16);
axis equal tight;
set(gca, 'YDir', 'reverse');  % Reverse Y-axis to match image coordinates

% Save the figure
output_file = '../bos_quiver_plot.png';
fprintf('Saving plot to %s...\n', output_file);
saveas(gcf, output_file);

fprintf('EXAMPLE_09_BOS_SIMPLE... FINISHED in %.1f sec.\n\n', toc(tAll));
