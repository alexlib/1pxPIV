% Example 07 - Treat sequences of images; use single-pixel correlation to compute average velocity field

clear;
tAll = tic;

%% Set processing parameters

% this variable defines, from where images should be read
imagePath = ['../../TestImages'];

% define pathe, where processing results will be stored
pivPar.spTargetPath = ['../../TestImages spOut v1'];

% Define image mask
pivPar.imMask1 = ['../../TestImages/Mask.bmp'];
pivPar.imMask2 = ['../../TestImages/Mask.bmp'];

% define the maximum displacements, for which the cross-correlation is evaluated
pivPar.spDeltaXNeg = 4;
pivPar.spDeltaXPos = 4;
pivPar.spDeltaYNeg = 4;
pivPar.spDeltaYPos = 16;   % try re-run with pivPar.spDeltaYPos = 20 for better result
% define binding (decreases resolution, but averages cross-correlation function from neighbouring pixels,
% improving accuracy)
pivPar.spBindX = 4;
pivPar.spBindY = 8;
pivPar.spStepX = 4;
pivPar.spStepY = 8;


% set remaining parameters to defaults
pivPar = pivParams([],pivPar,'defaults1Px');


%% Get list of images
aux1 = dir([imagePath, filesep, '*0.BMP']);
for kk = 1:numel(aux1)
    fileList1{kk} = [imagePath, filesep, aux1(kk).name];  %#ok<SAGROW>
end
fileList1 = sort(fileList1);
aux2 = dir([imagePath, filesep, '*1.BMP']);
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
plot(pivData1Px.spX(25,:),-pivData1Px.spV0(80,:),'r.');
hold on;
plot(pivData1Px.spX(25,:),-pivData1Px.spVfit(80,:),'b.');
xlabel('x (px)')
ylabel('displacement (px)');
legend('V from peak maximum','V from peak fit','Location','SouthEast');

%%
% plot the cross-correlation function
figure(2);
% following line will show cross-correlation function for pixels 0<X<300 and 650<Y<850. Observe sharp peaks on the left
% centered around zero displacement (low velocity low velocity fluctuation); peaks displaced downward in the center of
% jet; and peak with shape of inclined ellipsoid (shear layer with Reynolds stress).
pivQuiver(pivData1Px,'crop',[0,300,650,850],'spCC');


