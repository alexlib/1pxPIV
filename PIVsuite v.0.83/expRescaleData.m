function [pivData] = expRescaleData(pivPar,pivData)

im1 = pivData.imFilename1;
im2 = pivData.imFilename2;

% extract time information
Time1 = zeros(numel(im1),1)+NaN;
Time2 = zeros(numel(im1),1)+NaN;
for KI = 1:numel(im1)-1
    auxI1a = strfind(im1{KI},'\');    
    auxI1b = strfind(im1{KI},'/');
    auxI1 = sort([auxI1a,auxI1b]);
    auxI1 = auxI1(end)+1;
    auxStr = im1{KI}(auxI1:end);
    auxY = str2double(auxStr(1:2));
    auxM = str2double(auxStr(3:4));
    auxD = str2double(auxStr(5:6));
    auxh = str2double(auxStr(8:9));
    auxm = str2double(auxStr(10:11));
    auxs = str2double(auxStr(12:15));
    Time1(KI) = datenum(auxY,auxM,auxD,auxh,auxm,auxs);
    auxI1a = strfind(im1{KI},'\');    
    auxI1b = strfind(im1{KI},'/');
    auxI1 = sort([auxI1a,auxI1b]);
    auxI1 = auxI1(end)+1;
    auxStr = im2{KI}(auxI1:end);
    auxY = str2double(auxStr(1:2));
    auxM = str2double(auxStr(3:4));
    auxD = str2double(auxStr(5:6));
    auxh = str2double(auxStr(8:9));
    auxm = str2double(auxStr(10:11));
    auxs = str2double(auxStr(12:15));
    Time2(KI) = datenum(auxY,auxM,auxD,auxh,auxm,auxs);
end

% convert time in days to time in seconds, measured from the first image
Time0 = Time1(1);
Time1 = Time1-Time0;
Time2 = Time2-Time0;
Time1 = Time1 * 24*60*60;
Time2 = Time2 * 24*60*60;

% convert positions to scaled positions
pivData.X = pivData.X * pivPar.expScale;
pivData.Y = pivData.Y * pivPar.expScale;
pivData.iaSizeX = pivData.iaSizeX * pivPar.expScale;
pivData.iaSizeY = pivData.iaSizeY * pivPar.expScale;
pivData.iaStepX = pivData.iaStepX * pivPar.expScale;
pivData.iaStepY = pivData.iaStepY * pivPar.expScale;

% convert velocity to scaled velocity
for KI = 1:size(pivData.U,3)-1
    pivData.U(:,:,KI) = pivData.U(:,:,KI) * pivPar.expScale/(Time2(KI)-Time1(KI));
    pivData.V(:,:,KI) = pivData.V(:,:,KI) * pivPar.expScale/(Time2(KI)-Time1(KI));
end
pivData.U(:,:,end) = pivData.U(:,:,end) * pivPar.expScale/(Time2(end)-Time1(end));
pivData.V(:,:,end) = pivData.V(:,:,end) * pivPar.expScale/(Time2(end)-Time1(end));

% write time info
pivData.T = 1/2 * (Time1+Time2);

