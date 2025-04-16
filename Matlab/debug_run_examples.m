% Script to run all examples in PIVsuite with debugging
cd('PIVsuite v.0.8.3');
addpath(genpath('.'));

% Create a log file
global logFile;
logFile = fopen('../example_debug_log.txt', 'w');
fprintf(logFile, 'PIVsuite Examples Debug Log\n');
fprintf(logFile, 'Date: %s\n\n', datestr(now));

% List of examples to run
examples = {
    'example_01_Image_pair_simple.m',
    'example_02_Image_pair_standard.m',
    'example_03_Image_pair_advanced.m',
    'example_04_PIV_Challenge_A4.m',
    'example_05_Sequence_simple.m',
    'example_06a_Sequence_fast_and_on_drive.m',
    'example_06b_Sequence_multiprocessor.m',
    'example_07_PIV_Challenge_A3.m',
    'example_07_Sequence_SinglePix.m',
    'example_08a_PIV_Challenge_A1.m',
    'example_08b_PIV_Challenge_A2.m'
};

% Run each example
for i = 1:length(examples)
    exampleName = examples{i};

    global logFile;
    fprintf(logFile, '\n\n======================================================\n');
    fprintf(logFile, 'Running %s\n', exampleName);
    fprintf(logFile, '======================================================\n\n');

    fprintf('\n\n======================================================\n');
    fprintf('Running %s\n', exampleName);
    fprintf('======================================================\n\n');

    try
        % Check if the file exists
        if ~exist(exampleName, 'file')
            error('Example file %s does not exist', exampleName);
        end

        % Get the example name without extension
        [~, example_name, ~] = fileparts(exampleName);

        % Run the example with detailed error catching
        evalin('base', ['dbstop if error; ' example_name '; dbclear all;']);

        % Log success
        fprintf(logFile, 'Example %s completed successfully\n', exampleName);
        fprintf('Example %s completed successfully\n', exampleName);
    catch e
        % Log the error
        fprintf(logFile, '\nError running %s: %s\n', exampleName, e.message);
        fprintf(logFile, 'Stack trace:\n');
        fprintf(logFile, '%s\n', getReport(e));

        fprintf('\nError running %s: %s\n', exampleName, e.message);
        fprintf('Stack trace:\n');
        disp(getReport(e));
    end

    % Close all figures
    close all;

    % Pause to allow user to see results
    fprintf('\nFinished %s. Continuing in 1 second...\n', exampleName);
    pause(1);
end

% Close the log file
fclose(logFile);

fprintf('\nAll examples completed! Check example_debug_log.txt for details.\n');
