% Example 03 - advanced use of PIVsuite for obtaining the velocity field from a pair of images

clear;

%% 1. Defina image pair and image mask. Read data obtained by software DynamicStudio v. 2.10
im1 = ['..',filesep,'Data',filesep,'Test von Karman',filesep,'PIVlab_Karman_01.bmp'];
im2 = ['..',filesep,'Data',filesep,'Test von Karman',filesep,'PIVlab_Karman_02.bmp'];
imMask = ['..',filesep,'Data',filesep,'Test von Karman',filesep,'PIVlab_Karman_mask.png'];
load(['..',filesep,'Data',filesep,'Test von Karman',filesep,'dantec_frames_01_02.mat']);    % results of treatment by Dantec

%% 2. Define evaluation pivParameters

% initialize pivParameters and results
% pivPar = [];      % initialize treatment pivParameters
pivData = [];     % initialize detailed results that will go to this structure

% PARAMETERS DEFINING INTERROGATION AREAS
% masking image: can be different for each image. If no mask, set either [] or ''.
pivPar.imMask1 = imMask;
pivPar.imMask2 = imMask;
        % remarks: 
        % - image mask can be different for each image. Mask can be % either an image (with non-zero 
        % values for masked pixels) of a path to such an image. 
        % - If no mask, set either [] or ''. 
        % - If an image sequence is processed and a different mask is required for each frame, set imMask as 
        % a cellstring with paths to masking images.
        % - Masked pixels are replaced by mean pixel value within each interrogation area in expanded images
        % (see pivInterrogate.m). Owing to that, movement of the mask should not create any correlation peak.
% Size and overlap of interrogation areas
pivPar.anNpasses = 4;                % number of passes of PIV analysis
pivPar.iaSizeX = [64 32 16 20];     % x-size of interrogation areas in individual passes
pivPar.iaSizeY = [64 32 24 20];     % y-size of interrogation areas in individual passes
pivPar.iaStepX = [32 16  8 10];     % x-interval between interrogation areas in individual passes
pivPar.iaStepY = [32 16  8 10];     % y-interval between interrogation areas in individual passes
% Image deformation methodd: 
pivPar.iaMethod = 'defspline';      % 'basic' for clipping the image; 'offset' for moving interrogation areas
        % by an integer number of pixels; 'deflinear' for linear interpolation of displacement between a
        % velocity estimate; 'defspline' for spline interpolation of the displacement
pivPar.iaImageInterpolationMethod = 'linear';    % Defines, how images are interpolated. Possible values are 
        % 'linear' and 'spline'. Applies only to iaMethod 'deflinear' or 'defspline'.
pivPar.iaImageToDeform = 'Image1';  % defines, which image is deformed. Possible values are 'image1', 'image2',
        % 'both'.         
         
% PARAMETERS AFFECTING CROSS-CORRELATION
pivPar.ccMaxDisplacement = 0.9;    % If cross-correlation peak is further than ccMaxDisplacement * iaSize, 
        % it is ignored and a failure flag is set in the corresponding status
pivPar.ccRemoveIAMean = 1;         % before cross-correlating, ccRemoveIAMean * (mean pixel value of a given 
        % interrogatio area) is subtracted from the interrogation area. Value different from 1 is not
        % recommended.

% PARAMETERS AFFECTING VECTOR-FIELD VALIDATION
pivPar.vlDist = [1 1 2 2];                   % median kernel for validation will be 3x3 in first two passes,
                                             % then 5x5
pivPar.vlPasses = [2 2 1 1] ;                % number of passes by median-filter validation
pivPar.vlTresh = [2 2 2 1.5];                % median-filter-validation - acceptance treshold
pivPar.vlEps = 0.06;                         % usual noise level present in the velocity field [px]

% PARAMETERS AFFECTING REPLACEMENT OF SPURIOUS VECTORS
pivPar.rpMethod = 'inpaintT';                % method used for replacement of spurious vectors. For most cases, 
       % 'inpaint' and 'inpaintT' are appropriate, invoking John D'Enricco's inpaint_nan subroutine; the
       % latter is suitable for treating image sequences, as it takes in account also temporal evolution. For
       % other possible options, see comments in pivAnalyseImagePair.m, pivAnalyseImageSequence.m or
       % pivParams.m.

% PARAMETERS AFFECTING SMOOTHING OF THE VELOCITY FIELD
pivPar.smMethod = 'smoothn';                 % Smoothing method. 'Smoothn'
pivPar.smSigma = [1 1 0.5 0.3];

% PARAMETERS AFFECTING PLOTTING OF THE COMPUTED VELOCITY FIELD
pivPar.qvOptionsPair = {...                  % define plot shown between iterations
    'Umag',...                                          
    'quiver','selectStat','valid','linespec','-k',...    
    'quiver','selectStat','replaced','linespec','-w'};

% Hint: examine content of structure pivPar now (type "pivPar" to Matlab command line)


%% 3. Run the analysis
figure(1);   pause(0.1);    % open figure for plotting intermediate results
[pivData] = pivAnalyzeImagePair(im1,im2,pivData,pivPar);

fprintf('Elapsed time %.2f s (last pass %.2f s), subpixel interpolation failed for %.2f%% vectors, %.2f%% of vectors rejected.\n', ...
    sum(pivData.infCompTime), pivData.infCompTime(end), ...
    pivData.ccSubpxFailedN/pivData.N*100, pivData.spuriousN/pivData.N*100);

% Hint: examine content of structure pivData to see what is the structure of results of PIV analysis (type 
% "pivData" to Matlab command line)


%% 4. Show results

% A. Show velocity magnitude and quiver
figure(1);
hold off;
pivQuiver(pivData,...
    'Umag',...                                          % show background with magnitude 
    'quiver','selectStat','valid','linespec','-k',...   % show quiver with valid vectors shown by black 
    'quiver','selectStat','replaced','linespec','-w');  % show quiver with replaced vectors shown by white
title('Velocity magnitude (background) and velocity vectors (black: valid, white: replaced)');
xlabel('X [px]');
ylabel('Y [px]');

% B. Show velocity field as seen by observer moving together with the still fluid
% get median horizontal velocity
U = pivData.U;
U = U(~isnan(U));
U = median(U);
figure(2);
hold off;
pivQuiver(pivData,...
    'Umag','subtractU',U,'clipLo',0.1,...                  % show background with magnitude 
    'quiver','subtractU',U,'linespec','-k','qscale',-0.7);  % show quiver with replaced vectors shown by white
title('Velocity magnitude (background) and velocity vectors, as seen by observer moving together with the fluid');
xlabel('X [px]');
ylabel('Y [px]');

% C. Show some velocity profiles (at centers of interrogation areas):
% define points, for which velocity should be obtained
IX = [20,35,50,65,80,95];
linespec = {'-k.','-r.','-c.','-g.','-m.','-b.'};
% plot them and show also in Fig.1, where these profiles are
figure(3);
hold off;
for ki = 1:numel(IX)
    x = pivData.X(:,IX(ki));
    y = pivData.Y(:,IX(ki));
    u = pivData.U(:,IX(ki))-U;
    figure(3);
    plot(u,y,linespec{ki});
    hold on;
    figure(2);
    hold on;
    plot(x,y,linespec{ki});
    legendstr{ki} = ['x = ', num2str(x(1),'%.1f px')]; %#ok<SAGROW>
end
figure(3);
title('Velocity profiles');
legend(legendstr);
xlabel('x velocity component [px/frame]');
ylabel('Y [px]');

% D. Show velocity around the cylinder; velocity is interpolated to locations, in which the velocity is not
% measured
theta = 0:2*pi/50:2*pi;
x = 750 + (51+12)*cos(theta);
y = 385 + (51+12)*sin(theta);
u = interp2(pivData.X,pivData.Y,pivData.U,x,y)-U;
v = interp2(pivData.X,pivData.Y,pivData.V,x,y);
figure(2);
plot(x,y,'-r.');
figure(4);
hold off;
plot(theta,sqrt(u.^2+v.^2),'-r.');
title('Velocity magnitude around the cylinder');
xlabel('\theta [rad]');
ylabel('U_{mag} [px/frame]');

