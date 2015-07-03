% Example 07 - Treat sequences of images; use single-pixel correlation to compute average velocity field

clear;
tAll = tic;

%% Set processing parameters

% this variable defines, from where images should be read
imagePath = ['../../M03 Source imgs'];

% define pathe, where processing results will be stored
pivPar.spTargetPath = ['../../M03 spOut v1'];

% Define image mask
pivPar.imMask1 = ['../../M03 Source imgs/Mask.bmp'];
pivPar.imMask2 = ['../../M03 Source imgs/Mask.bmp'];

% define the maximum displacements, for which the cross-correlation is evaluated
pivPar.spDeltaXNeg = 4;
pivPar.spDeltaXPos = 4;
pivPar.spDeltaYNeg = 11;
pivPar.spDeltaYPos = 4;
% define binding (decreases resolution, but averages cross-correlation function from neighbouring pixels,
% improving accuracy)
pivPar.spBindX = 4;
pivPar.spBindY = 16;
pivPar.spStepX = 4;
pivPar.spStepY = 16;


% set remaining parameters to defaults
pivPar = pivParams([],pivPar,'defaults1Px');


%% Get list of images
aux1 = dir([imagePath, filesep, '*a.bmp']);
for kk = 1:numel(aux1)
    fileList1{kk} = [imagePath, filesep, aux1(kk).name];  %#ok<SAGROW>
end
fileList1 = sort(fileList1);
aux2 = dir([imagePath, filesep, '*b.bmp']);
for kk = 1:numel(aux2)
    fileList2{kk} = [imagePath, filesep, aux2(kk).name];  %#ok<SAGROW>
end
fileList2 = sort(fileList2);


%% do 1pxPIV analysis
pivData1Px = [];
pivData1Px = pivSinglepixAnalyze(fileList1,fileList2,pivData1Px,pivPar);


% save results
save([pivPar.spTargetPath filesep 'pivData.mat'],'pivData1Px');


%% Plot some results
% plot the vertical velocity component - a "color map"
figure(1);
subplot(2,1,1);
pivQuiver(pivData1Px,'spV0');

% plot profile of vertical velocity in the middle
subplot(2,1,2);
hold off;
plot(pivData1Px.spX(25,:),-pivData1Px.spV0(25,:),'r.');
hold on;
plot(pivData1Px.spX(25,:),-pivData1Px.spVfit(25,:),'b.');
xlabel('x (px)')
ylabel('displacement (px)');

% plot std of local mean velocity - just to indicate the same velocity is found regardless Y
plot(pivData1Px.spX(25,:),std(pivData1Px.spV0,1),'r-');
legend('V from peak maximum','V from peak fit','V_{rms}','Location','SouthEast');

%%
% plot the cross-correlation function
figure(2);
pivQuiver(pivData1Px,'crop',[35,135,360,440],'spCC');


%%
% plot <u'u'>, <v'v'> and <u'v'> profiles
figure(2);
subplot(2,1,1);
pivQuiver(pivData1Px,'spRSuv');