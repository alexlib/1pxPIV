% Basic test script for PIV analysis using MATLAB PIVSuite
% This script creates a pair of synthetic images with a known displacement
% and tests a simplified PIV analysis function.

% Create synthetic images
[im1, im2] = createSyntheticImages(128, 0.01, 2.0, 5.0);

% Analyze image pair
[X, Y, u, v] = basicPivAnalysis(im1, im2, 32, 16);

% Compute error
u_error = mean(abs(u(:) - 5.0));
v_error = mean(abs(v(:)));

fprintf('Mean absolute error in u: %.6f\n', u_error);
fprintf('Mean absolute error in v: %.6f\n', v_error);

% Create output directory if it doesn't exist
if ~exist('../output', 'dir')
    mkdir('../output');
end

% Plot results
figure('Position', [100, 100, 1200, 800]);

% Plot synthetic images
subplot(2, 2, 1);
imagesc(im1);
colormap(gca, gray);
title('Synthetic Image 1');
axis off;

subplot(2, 2, 2);
imagesc(im2);
colormap(gca, gray);
title('Synthetic Image 2');
axis off;

% Plot displacement field
subplot(2, 2, 3);
quiver(X, Y, u, v, 'AutoScale', 'off');
title('Displacement Field');
axis equal;

% Plot error
subplot(2, 2, 4);
histogram(u(:), 20);
hold on;
histogram(v(:), 20);
xline(5.0, 'r--', 'True u');
xline(0.0, 'g--', 'True v');
title('Displacement Histogram');
legend('u', 'v', 'True u', 'True v');

% Save figure
saveas(gcf, '../output/matlab_basic_piv_test.png');

% Check if the error is acceptable
if u_error < 0.5 && v_error < 0.5
    fprintf('Test PASSED: Error is within acceptable limits.\n');
else
    fprintf('Test FAILED: Error is too large.\n');
end

% Function to create synthetic images
function [im1, im2] = createSyntheticImages(size, particle_density, particle_size, displacement)
    % Create a grid
    [X, Y] = meshgrid(1:size, 1:size);
    
    % Create particles
    n_particles = round(size * size * particle_density);
    particles_x = randi(size, n_particles, 1);
    particles_y = randi(size, n_particles, 1);
    
    % Create the first image
    im1 = zeros(size, size);
    for i = 1:n_particles
        % Add a Gaussian particle
        im1 = im1 + exp(-((X - particles_x(i)).^2 + (Y - particles_y(i)).^2) / (2 * particle_size^2));
    end
    
    % Normalize the image
    im1 = im1 / max(im1(:));
    
    % Create the second image
    im2 = zeros(size, size);
    for i = 1:n_particles
        % Add a Gaussian particle with displacement
        im2 = im2 + exp(-((X - particles_x(i) - displacement).^2 + (Y - particles_y(i)).^2) / (2 * particle_size^2));
    end
    
    % Normalize the image
    im2 = im2 / max(im2(:));
end

% Function to perform basic PIV analysis
function [X, Y, u, v] = basicPivAnalysis(im1, im2, window_size, step_size)
    % Get image dimensions
    [im_size_y, im_size_x] = size(im1);
    
    % Calculate number of interrogation areas
    ia_n_x = floor((im_size_x - window_size) / step_size) + 1;
    ia_n_y = floor((im_size_y - window_size) / step_size) + 1;
    
    % Create grid of interrogation area centers
    x = window_size/2 + (0:ia_n_x-1) * step_size;
    y = window_size/2 + (0:ia_n_y-1) * step_size;
    [X, Y] = meshgrid(x, y);
    
    % Initialize arrays for displacement
    u = zeros(ia_n_y, ia_n_x);
    v = zeros(ia_n_y, ia_n_x);
    
    % Process each interrogation area
    for iy = 1:ia_n_y
        for ix = 1:ia_n_x
            % Get the interrogation area from the first image
            ia1_y = (iy-1) * step_size + 1;
            ia1_x = (ix-1) * step_size + 1;
            ia1 = im1(ia1_y:ia1_y+window_size-1, ia1_x:ia1_x+window_size-1);
            
            % Get the interrogation area from the second image
            ia2 = im2(ia1_y:ia1_y+window_size-1, ia1_x:ia1_x+window_size-1);
            
            % Remove mean
            ia1 = ia1 - mean(ia1(:));
            ia2 = ia2 - mean(ia2(:));
            
            % Compute cross-correlation using FFT
            corr = fftshift(real(ifft2(conj(fft2(ia1)) .* fft2(ia2))));
            
            % Find the peak
            [~, max_idx] = max(corr(:));
            [peak_y, peak_x] = ind2sub(size(corr), max_idx);
            
            % Compute displacement (peak position relative to center)
            dx = peak_x - window_size/2 - 0.5;
            dy = peak_y - window_size/2 - 0.5;
            
            % Store displacement
            u(iy, ix) = dx;
            v(iy, ix) = dy;
            
            % Sub-pixel refinement using Gaussian peak fit
            if peak_x > 1 && peak_x < window_size && peak_y > 1 && peak_y < window_size
                % Fit Gaussian in x direction
                c1 = log(corr(peak_y, peak_x-1));
                c2 = log(corr(peak_y, peak_x));
                c3 = log(corr(peak_y, peak_x+1));
                if c1 < c2 && c3 < c2  % Check if it's a peak
                    dx_sub = 0.5 * (c1 - c3) / (c1 - 2*c2 + c3);
                    u(iy, ix) = u(iy, ix) + dx_sub;
                end
                
                % Fit Gaussian in y direction
                c1 = log(corr(peak_y-1, peak_x));
                c2 = log(corr(peak_y, peak_x));
                c3 = log(corr(peak_y+1, peak_x));
                if c1 < c2 && c3 < c2  % Check if it's a peak
                    dy_sub = 0.5 * (c1 - c3) / (c1 - 2*c2 + c3);
                    v(iy, ix) = v(iy, ix) + dy_sub;
                end
            end
        end
    end
end
