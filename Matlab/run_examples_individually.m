% Script to run each example individually with error handling
cd('PIVsuite v.0.8.3');
addpath(genpath('.'));

% Create a log file
logFile = fopen('../example_debug_log.txt', 'w');
fprintf(logFile, 'PIVsuite Examples Debug Log\n');
fprintf(logFile, 'Date: %s\n\n', datestr(now));

% Run example 1
run_single_example('example_01_Image_pair_simple.m');

% Run example 2
run_single_example('example_02_Image_pair_standard.m');

% Run example 3
run_single_example('example_03_Image_pair_advanced.m');

% Run example 4
run_single_example('example_04_PIV_Challenge_A4.m');

% Run example 5
run_single_example('example_05_Sequence_simple.m');

% Run example 6a
run_single_example('example_06a_Sequence_fast_and_on_drive.m');

% Run example 6b
run_single_example('example_06b_Sequence_multiprocessor.m');

% Run example 7 (PIV Challenge A3)
run_single_example('example_07_PIV_Challenge_A3.m');

% Run example 7 (Sequence SinglePix)
run_single_example('example_07_Sequence_SinglePix.m');

% Run example 8a
run_single_example('example_08a_PIV_Challenge_A1.m');

% Run example 8b
run_single_example('example_08b_PIV_Challenge_A2.m');

% Close the log file
fclose(logFile);

fprintf('\nAll examples completed! Check example_debug_log.txt for details.\n');
