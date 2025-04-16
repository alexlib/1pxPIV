% Script to check data paths for PIVsuite examples
fprintf('Checking data paths for PIVsuite examples...\n\n');

% Check if the PIVsuite directory exists
if exist('PIVsuite v.0.8.3', 'dir')
    fprintf('PIVsuite v.0.8.3 directory found.\n');
else
    fprintf('ERROR: PIVsuite v.0.8.3 directory not found!\n');
end

% Check if the Data directory exists
if exist('Data', 'dir')
    fprintf('Data directory found.\n');
else
    fprintf('ERROR: Data directory not found!\n');
end

% Check for specific data directories
dataDirs = {
    'Data/Test Tububu',
    'Data/PIV Challenge 2005 A1',
    'Data/PIV Challenge 2005 A2',
    'Data/PIV Challenge 2005 A3',
    'Data/PIV Challenge 2005 A4'
};

for i = 1:length(dataDirs)
    if exist(dataDirs{i}, 'dir')
        fprintf('Directory %s found.\n', dataDirs{i});
        
        % Count image files in this directory
        files = dir(fullfile(dataDirs{i}, '*.bmp'));
        fprintf('  - Contains %d BMP files.\n', length(files));
    else
        fprintf('WARNING: Directory %s not found!\n', dataDirs{i});
    end
end

% Check for example files
cd('PIVsuite v.0.8.3');
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

fprintf('\nChecking example files:\n');
for i = 1:length(examples)
    if exist(examples{i}, 'file')
        fprintf('Example file %s found.\n', examples{i});
    else
        fprintf('ERROR: Example file %s not found!\n', examples{i});
    end
end

% Check for required functions
requiredFunctions = {
    'pivParams.m',
    'pivAnalyzeImagePair.m',
    'pivSinglepixAnalyze.m',
    'pivCreateImageSequence.m'
};

fprintf('\nChecking required functions:\n');
for i = 1:length(requiredFunctions)
    if exist(requiredFunctions{i}, 'file')
        fprintf('Function %s found.\n', requiredFunctions{i});
    else
        fprintf('ERROR: Function %s not found!\n', requiredFunctions{i});
    end
end

fprintf('\nData path check completed.\n');
