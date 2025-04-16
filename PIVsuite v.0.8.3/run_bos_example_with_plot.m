% Script to run the BOS example and create a quiver plot over the image
cd('PIVsuite v.0.8.3');

% Run the BOS example
try
    % First, modify the example to save the results
    example_code = fileread('example_09_BOS_image_pair.m');
    
    % Add code to save the results before the end of the script
    save_code = sprintf('\n%% Save the results to a .mat file for later analysis\nresultFile = ''bos_example_results.mat'';\nfprintf(''Saving results to %%s...\\n'', resultFile);\nsave(resultFile, ''pivData'', ''im1'', ''im2'', ''pivPar'');\n\n');
    
    % Find the position to insert the save code (before the final fprintf)
    final_fprintf_pos = strfind(example_code, 'fprintf(''EXAMPLE_09_BOS_IMAGE_PAIR... FINISHED');
    if ~isempty(final_fprintf_pos)
        % Insert the save code before the final fprintf
        modified_code = [example_code(1:final_fprintf_pos-1), save_code, example_code(final_fprintf_pos:end)];
        
        % Write the modified code to a temporary file
        temp_file = 'temp_example_09.m';
        fid = fopen(temp_file, 'w');
        fprintf(fid, '%s', modified_code);
        fclose(fid);
        
        % Run the modified example
        fprintf('Running modified BOS example...\n');
        run(temp_file);
        
        % Clean up the temporary file
        delete(temp_file);
    else
        % If we couldn't find the insertion point, just run the original example
        fprintf('Running original BOS example...\n');
        example_09_BOS_image_pair;
    end
    
    % Load the results if they were saved
    result_file = 'bos_example_results.mat';
    if exist(result_file, 'file')
        fprintf('Loading results from %s...\n', result_file);
        load(result_file);
    end
    
    % Create a quiver plot over the image
    fprintf('Creating quiver plot over the image...\n');
    
    % Create a new figure
    figure('Position', [100, 100, 1200, 900]);
    
    % Display the image
    imagesc(im1);
    colormap gray;
    hold on;
    
    % Extract the velocity vectors and coordinates
    x = pivData.x;
    y = pivData.y;
    u = pivData.u;
    v = pivData.v;
    
    % Downsample the vectors for better visualization
    step = 8;  % Adjust this value to change the density of arrows
    
    % Plot the quiver
    quiver(x(1:step:end, 1:step:end), y(1:step:end, 1:step:end), ...
           u(1:step:end, 1:step:end), v(1:step:end, 1:step:end), ...
           'r', 'LineWidth', 1.5, 'AutoScale', 'on', 'AutoScaleFactor', 2);
    
    % Add title and adjust the plot
    title('BOS Image with Velocity Field', 'FontSize', 16);
    axis equal tight;
    set(gca, 'YDir', 'reverse');  % Reverse Y-axis to match image coordinates
    
    % Save the figure
    output_file = '../bos_quiver_plot.png';
    fprintf('Saving plot to %s...\n', output_file);
    saveas(gcf, output_file);
    
    % Also create a figure with just the quiver plot
    figure('Position', [100, 100, 1200, 900]);
    quiver(x(1:step:end, 1:step:end), y(1:step:end, 1:step:end), ...
           u(1:step:end, 1:step:end), v(1:step:end, 1:step:end), ...
           'k', 'LineWidth', 1.5, 'AutoScale', 'on', 'AutoScaleFactor', 2);
    title('Velocity Field (Quiver Plot)', 'FontSize', 16);
    axis equal;
    grid on;
    
    % Save the quiver-only figure
    output_file = '../bos_quiver_only.png';
    fprintf('Saving quiver-only plot to %s...\n', output_file);
    saveas(gcf, output_file);
    
    fprintf('Done!\n');
catch e
    fprintf('Error: %s\n', e.message);
    disp(getReport(e));
end

% Return to the original directory
cd('..');
