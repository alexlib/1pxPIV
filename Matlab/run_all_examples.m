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

disp('Running example_06b_Sequence_multiprocessor...');
try
    example_06b_Sequence_multiprocessor;
    disp('Example 6b completed successfully');
catch e
    disp(['Error in Example 6b: ' e.message]);
end
close all;
pause(1);

disp('Running example_07_PIV_Challenge_A3...');
try
    example_07_PIV_Challenge_A3;
    disp('Example 7 (PIV Challenge A3) completed successfully');
catch e
    disp(['Error in Example 7 (PIV Challenge A3): ' e.message]);
end
close all;
pause(1);

disp('Running example_07_Sequence_SinglePix...');
try
    example_07_Sequence_SinglePix;
    disp('Example 7 (Sequence SinglePix) completed successfully');
catch e
    disp(['Error in Example 7 (Sequence SinglePix): ' e.message]);
end
close all;
pause(1);

disp('Running example_08a_PIV_Challenge_A1...');
try
    example_08a_PIV_Challenge_A1;
    disp('Example 8a completed successfully');
catch e
    disp(['Error in Example 8a: ' e.message]);
end
close all;
pause(1);

disp('Running example_08b_PIV_Challenge_A2...');
try
    example_08b_PIV_Challenge_A2;
    disp('Example 8b completed successfully');
catch e
    disp(['Error in Example 8b: ' e.message]);
end
close all;
pause(1);

disp('Running example_09_BOS_image_pair...');
try
    example_09_BOS_image_pair;
    disp('Example 9 (BOS image pair) completed successfully');
catch e
    disp(['Error in Example 9 (BOS image pair): ' e.message]);
end
close all;

disp('All examples completed!');
