% Script to run a single example with error handling
function run_single_example(exampleName)
    fprintf('\n\n======================================================\n');
    fprintf('Running %s\n', exampleName);
    fprintf('======================================================\n\n');
    
    try
        % Get the example name without extension
        [~, example_name, ~] = fileparts(exampleName);
        
        % Run the example
        run(example_name);
        
        fprintf('\nExample %s completed successfully\n', exampleName);
    catch e
        fprintf('\nError running %s: %s\n', exampleName, e.message);
        fprintf('Stack trace:\n');
        disp(getReport(e));
    end
    
    % Close all figures
    close all;
    
    % Pause to allow user to see results
    fprintf('\nFinished %s. Press Enter to continue...\n', exampleName);
    pause(1);
end
