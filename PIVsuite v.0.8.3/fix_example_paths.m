% Script to fix paths in PIVsuite examples
fprintf('Fixing paths in PIVsuite examples...\n\n');

% Define path mappings (expected path -> actual path)
pathMappings = {
    {'../Data/PIV Challenge 2005 A1', '../Data/Test PIVChallenge3A1'},
    {'../Data/PIV Challenge 2005 A2', '../Data/Test PIVChallenge3A2'},
    {'../Data/PIV Challenge 2005 A3', '../Data/Test PIVChallenge3A3'},
    {'../Data/PIV Challenge 2005 A4', '../Data/Test PIVChallenge3A4'}
};

% Change to PIVsuite directory
cd('PIVsuite v.0.8.3');

% List of example files to fix
exampleFiles = {
    'example_04_PIV_Challenge_A4.m',
    'example_07_PIV_Challenge_A3.m',
    'example_08a_PIV_Challenge_A1.m',
    'example_08b_PIV_Challenge_A2.m'
};

% Fix each example file
for i = 1:length(exampleFiles)
    fileName = exampleFiles{i};
    fprintf('Fixing paths in %s...\n', fileName);
    
    % Read the file content
    fid = fopen(fileName, 'r');
    if fid == -1
        fprintf('ERROR: Could not open file %s\n', fileName);
        continue;
    end
    
    content = '';
    line = fgetl(fid);
    while ischar(line)
        % Check if this line contains a path that needs to be fixed
        for j = 1:length(pathMappings)
            oldPath = pathMappings{j}{1};
            newPath = pathMappings{j}{2};
            
            % Replace the path if found
            line = strrep(line, oldPath, newPath);
        end
        
        % Add the line to the content
        content = [content, line, newline];
        
        % Read the next line
        line = fgetl(fid);
    end
    
    fclose(fid);
    
    % Write the modified content back to the file
    fid = fopen(fileName, 'w');
    if fid == -1
        fprintf('ERROR: Could not open file %s for writing\n', fileName);
        continue;
    end
    
    fprintf(fid, '%s', content);
    fclose(fid);
    
    fprintf('Successfully fixed paths in %s\n', fileName);
end

fprintf('\nPath fixing completed.\n');
