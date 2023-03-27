%% Road quality classification by PCA based decision tree overlay using accelerometer and inertial sensors
% written by Roland Nagy, János Abonyi and Alex Kummer


clear all
close all
clc

%Loading data
filename='2022_05_16_ZEG';

load(append(filename,'_labels.mat'),'q')
opts = detectImportOptions(filename);                           
opts.Delimiter=';';
opts.DataLines=[1 Inf];
data=readtable(filename,opts);      
% Data order by column: 1-2 Accelerometer; 3-4-5 XYZ IMU Accelerometer,
% 6-7-8 XYZ IMU Gyro, 9-10 GPS coord., 11 Time
x=[];
for j=[1 2 5 6 7 8 9 10 11 12 13]
x=[x   str2double(string(data{:,j}))];
end

x(:,12) = q;

clear data opts j

% Loading classification model 
load('fittedDecisionTree.mat')

%% Signal processing

% Detrend
for j = [1 2]
    x(:,j) = highpass(x(:,j),0.001)+mean(x(:,j));
end
clear i j

% Interpolation of GPS data
z = [1:101:length(x)]';
xq = [1:length(x)]';
for j=[9 10 11]
    v = (x(1:101:end,j));
    x(:,j) = interp1(z,v,xq);
end
clear z xq v j

% Remove NaN values - 1
x(length(x)-102:length(x),:) = [];

% GPS data smoothing
x(:,9) = movmean(x(:,9),500);
x(:,10) = movmean(x(:,10),500);

% Remove NaN values - 2
x(length(x)-102:length(x),:) = [];

% Remove bias voltage
for j = [1 2 3 4 5 6 7 8]
    x(:,j) = x(:,j)-mean(x(:,j));
end
clear j

for j = [1 2]
    x(:,j) = highpass(x(:,j),0.001);
end
clear j

% Same scale
x(:,1) = rescale(x(:,1),-20,20);
x(:,2) = rescale(x(:,2),-20,20);
x(:,3) = rescale(x(:,3),-20,20);
x(:,4) = rescale(x(:,4),-20,20);
x(:,5) = rescale(x(:,5),-20,20);
x(:,6) = rescale(x(:,6),-20,20);
x(:,7) = rescale(x(:,7),-20,20);
x(:,8) = rescale(x(:,8),-20,20);

%% Resampling by spatial frequency

Lat = x(:,9);
Lon = x(:,10);

% Calculate distance between each data point
for j=[2:length(x)]
    d = abs(acos(cosd(90-Lat(j-1,1)) .* cosd(90-Lat((j),1))...
        + sind(90-Lat(j-1,1)) .* sind(90-Lat((j),1)))...
        .* cosd(Lon(j-1,1)-Lon((j),1)) * 3958.76);
    dDist(j,1) = d;
end
clear Lat Lon d j

% Resampling
x(:,13) = cumsum(dDist)*1000;
xq = [0:0.2:max(x(:,13))]';
[z, index] = unique(x(:,13));

for j = [1 2 3 4 5 6 7 8 9 10 11 12]
    v = x(:,j);
    x_sd(:,j) = interp1(z, v(index), xq);
end
clear index z v xq j

% Create geopointshape object for plotting results
Shape = geopointshape(x_sd(:,9),x_sd(:,10));

% Remove NaN values
dt_sd=diff(x_sd(:,11))/1000;
index=isnan(dt_sd);
dt_sd(index)=[];        
x_sd(index,:)=[];
x_sd(end,:)=[];
Shape(index,:)=[];
Shape(end,:)=[];
clear index j

%% Feature extraction

% Calculate spectrum
t=cumsum(dt_sd);
t(end)=t(end-1)+(t(end-1)-t(end-2));

Pv={};
Fv={};
for j=[1 2 3 4 5 6 7 8]
    xs=x_sd(:,j);
    TT=timetable(seconds(t),xs);
    [P,F,T]=pspectrum(TT,'spectrogram');
    figure
    pspectrum(TT,'spectrogram');
    sP=sum(P);
    xDesired = linspace(seconds(min(T)),seconds(max(T)),length(t))';
    Pv{j} = interp1(T,P',seconds(xDesired));
    Fv{j}=F;
end
clear j sP P Fv T TT xs

% Highlight 8-10 Hz range
lowL = round((1024/max(F))*8,0);
topL = round((1024/max(F))*10,0);

% Calculate mean in the highlighted range
for j=[1 2 3 4 5 6 7 8]
    for k = 1:size(Pv{1,j},1)
        Xt(k,j) = mean((Pv{j}(k,lowL:topL)));
    end
end

% PCA
[coefs,score,latent,tsquared,explained] = pca(pow2db(Xt(1:1:end,[1 2 3 4 5 6 7 8])));

%% Predict road quality classes and metrics

[predicted,score_predict] = predict(tree_pca_v2,score);
predictedFilt = round(movmean(predicted,450),0);

% Plot the results on a map
figure
GT = table(Shape,predictedFilt);
gmap = geoplot(GT,ColorVariable="predictedFilt",MarkerSize=15);
geobasemap streets-light
colorbar

% confusion matrix
Y = round(x_sd(:,12),0);
predicted = round(predicted,0);
Y = round(Y,0);
C = confusionmat(Y,predicted);
figure
cm = confusionchart(C,'RowSummary','row-normalized');

% meroszamok
rateOfClasses = tabulate(categorical(Y)); %osztalyok aranya
accuracey = sum(diag(C))/length(Y); %accuracy, foatlo aranya

%scree plot
figure
pareto(explained,1);
xlabel('Principal Component');
ylabel('Variance Explained (%)');

%biplot
figure
vbls = {'Acc1','Acc2','IMU acc x','IMU acc y','IMU acc z','IMU gyro x','IMU gyro y','IMU gyro z'}; % Labels for the variables
biplot(coefs(:,1:2),'Scores',score(:,1:2),'VarLabels',vbls);
