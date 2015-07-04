% Example 04 - Treat sequences of images; as simple as possible

clear;

%% Set processing parameters

% this variable defines, from where images should be read
imagePath = ['..',filesep,'Data',filesep,'Test Tububu'];

% Parameters for processing first pair of images
pivPar.iaSizeX = [64 32 16];                 % interrogation area size for first, second and third pass
pivPar.qvPair = {...                         % define plot shown between iterations
    'Umag','clipHi',3,...                                          
    'quiver','selectStat','valid','linespec','-k',...    
    'quiver','selectStat','replaced','linespec','-w'};
pivPar.seqPairInterval = 3;                  % process only one image pair on three
pivPar = pivParams([],pivPar,'defaultsSeq');


%% Analyze images

% get list of images
aux = dir([imagePath, filesep, '*.bmp']);
for kk = 1:numel(aux)
    fileList{kk} = [imagePath, filesep, aux(kk).name];  %#ok<SAGROW>
end
fileList = sort(fileList);

% do PIV analysis
figure(1);
[im1,im2] = pivCreateImageSequence(fileList,pivPar);
pivData = pivAnalyzeImageSequence(im1,im2,[],pivPar);

%% Show results

figure(1);
for kk=1:size(pivData.U,3)
    hold off;
    pivQuiver(pivData,'timeSlice',kk,...
        'Umag','clipLo',0,'clipHi',3,...
        'quiver','linespec','-k','selectStat','valid',...
        'quiver','linespec','-w','selectStat','replaced');
    title('Background: U_{mag}. Quiver: velocity (black: valid, white: replaced)');
    F(kk) = getframe;                    %#ok<SAGROW>
end

movie(F,3);
