% Script to run all examples in PIVsuite
cd('PIVsuite v.0.8.3');

disp('Running example_01_Image_pair_simple...');
try
    example_01_Image_pair_simple;
    disp('Example 1 completed successfully');
catch e
    disp(['Error in Example 1: ' e.message]);
end
close all;
pause(1);

disp('Running example_02_Image_pair_standard...');
try
    example_02_Image_pair_standard;
    disp('Example 2 completed successfully');
catch e
    disp(['Error in Example 2: ' e.message]);
end
close all;
pause(1);

disp('Running example_03_Image_pair_advanced...');
try
    example_03_Image_pair_advanced;
    disp('Example 3 completed successfully');
catch e
    disp(['Error in Example 3: ' e.message]);
end
close all;
pause(1);

disp('Running example_04_PIV_Challenge_A4...');
try
    example_04_PIV_Challenge_A4;
    disp('Example 4 completed successfully');
catch e
    disp(['Error in Example 4: ' e.message]);
end
close all;
pause(1);

disp('Running example_05_Sequence_simple...');
try
    example_05_Sequence_simple;
    disp('Example 5 completed successfully');
catch e
    disp(['Error in Example 5: ' e.message]);
end
close all;
pause(1);

disp('Running example_06a_Sequence_fast_and_on_drive...');
try
    example_06a_Sequence_fast_and_on_drive;
    disp('Example 6a completed successfully');
catch e
    disp(['Error in Example 6a: ' e.message]);
end
close all;
pause(1);

disp('Running example_07_PIV_Challenge_A3...');
try
    example_07_PIV_Challenge_A3;
    disp('Example 7 completed successfully');
catch e
    disp(['Error in Example 7: ' e.message]);
end
close all;

disp('All examples completed!');
