% Script to process BOS images and create a quiver plot
% This script uses a simpler approach without relying on PIVsuite functions

% Clear workspace
clear;
close all;

fprintf('Processing BOS images...\n');
tic;

% Define paths to images
im1Path = fullfile('Data', 'Test BOS', '11-49-28.000-4.tif');
im2Path = fullfile('Data', 'Test BOS', '11-49-28.000-6.tif');

fprintf('Image paths:\n  %s\n  %s\n', im1Path, im2Path);

% Check if images exist
if ~exist(im1Path, 'file') || ~exist(im2Path, 'file')
    error('BOS image files not found. Please check the paths.');
end

% Load images
fprintf('Loading images...\n');
im1 = imread(im1Path);
im2 = imread(im2Path);

% Convert to grayscale if needed
if size(im1, 3) > 1
    im1 = rgb2gray(im1);
end
if size(im2, 3) > 1
    im2 = rgb2gray(im2);
end

% Convert to double
im1 = double(im1);
im2 = double(im2);

% Downsample images for faster processing
scale = 0.25; % Process at 25% of original size
im1 = imresize(im1, scale);
im2 = imresize(im2, scale);

fprintf('Images loaded and resized. Size: %d x %d pixels\n', size(im1, 2), size(im1, 1));

% Set parameters for optical flow
windowSize = 32;
stepSize = 16;

% Calculate optical flow using Lucas-Kanade method
fprintf('Calculating optical flow...\n');

% Create grid of points - only from row 200 to the end
[height, width] = size(im1);

% Define the starting row (after resizing)
startRow = round(200 * scale);
fprintf('Starting analysis from row %d (scaled from original row 200)\n', startRow);

% Create grid only for the lower part of the image
[X, Y] = meshgrid(1:stepSize:width, startRow:stepSize:height);
points = [X(:), Y(:)];

% Initialize velocity vectors
u = zeros(size(X));
v = zeros(size(Y));

% Calculate optical flow for each point
for i = 1:size(points, 1)
    x = points(i, 1);
    y = points(i, 2);

    % Skip points too close to the edge
    if x <= windowSize/2 || x > width-windowSize/2 || y <= windowSize/2 || y > height-windowSize/2
        continue;
    end

    % Extract windows
    win1 = im1(y-windowSize/2:y+windowSize/2, x-windowSize/2:x+windowSize/2);
    win2 = im2(y-windowSize/2:y+windowSize/2, x-windowSize/2:x+windowSize/2);

    % Calculate gradient
    [Gx, Gy] = gradient(win1);

    % Calculate temporal gradient
    Gt = win2 - win1;

    % Reshape gradients to vectors
    Gx = Gx(:);
    Gy = Gy(:);
    Gt = Gt(:);

    % Create A matrix
    A = [Gx, Gy];

    % Calculate flow using least squares
    flow = -pinv(A' * A) * A' * Gt;

    % Store flow vectors
    row = ceil((y-1)/stepSize) + 1;
    col = ceil((x-1)/stepSize) + 1;
    if row <= size(u, 1) && col <= size(u, 2)
        u(row, col) = flow(1);
        v(row, col) = flow(2);
    end
end

fprintf('Optical flow calculated.\n');

% Create quiver plot over the image
fprintf('Creating quiver plot over the image...\n');

% Create a new figure
figure('Position', [100, 100, 1200, 900]);

% Display the image
imagesc(im1);
colormap gray;
hold on;

% Scale the vectors for better visualization
scale_factor = 55;  % Increased from 5 to 15 for stronger arrows
u_scaled = u * scale_factor;
v_scaled = v * scale_factor;

% Plot the quiver with stronger arrows
quiver(X, Y, u_scaled, v_scaled, 'r', 'LineWidth', 2.0, 'MaxHeadSize', 1.0);

% Add title and adjust the plot
title('BOS Image with Velocity Field - Lower Region', 'FontSize', 16);
axis equal tight;
set(gca, 'YDir', 'reverse');  % Reverse Y-axis to match image coordinates

% Save the figure
output_file = 'bos_quiver_plot_lower.png';
fprintf('Saving plot to %s...\n', output_file);
saveas(gcf, output_file);

% Also create a figure with just the quiver plot
figure('Position', [100, 100, 1200, 900]);
quiver(X, Y, u_scaled, v_scaled, 'k', 'LineWidth', 2.0, 'MaxHeadSize', 1.0);
title('Velocity Field (Quiver Plot - Lower Region)', 'FontSize', 16);
axis equal;
grid on;

% Save the quiver-only figure
output_file = 'bos_quiver_only_lower.png';
fprintf('Saving quiver-only plot to %s...\n', output_file);
saveas(gcf, output_file);

fprintf('Processing completed in %.1f seconds.\n', toc);
