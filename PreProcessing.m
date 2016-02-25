%% Accelerometer Data Preperation for Classification:
clear all; close all; 
init = load('PUC_withUser');
data = init.data;
target = init.target;
rng('default');
%%now 'data' contains all of the accelerometer data from the HAR data set:
% by column: 1) User ID 2)X1 3)Y1 4) Z1) 5)X2 6)Y2 7)Z2 8)X3 9)Y3 10)Z3
% 11)X4 12)Y4 13)Z4
%and 'target ' containts all of the class data (0-4).
% Class: 0) Sitting 1) Sitting Down 2) Standing 3) Standing Up 4) Walking
%%
%From each exercise of each person, calculate the variance of pitch and
% roll
idx = {};
wholedata = [data, target];
class_colors = {'r','b','g','cyan','yellow'};
class_names = {'Sitting','Sitting Down','Standing','Standing Up','Walking'};
i = 1;
figure();
for k=0:4
    figure(k+1);
    t = sprintf('Scatter Plot of 3-Axis Accelerometer Readings: %s',class_names{k+1});
    title(t);
    hold on;
    for user = 0:3
        [data_idx,I] = find(wholedata(:,1) == user & wholedata(:,14) == k);
        idx{k+1,user+1} = data_idx;
        i = i + 1;
        for j = 1:4
            scatter3(data(idx{k+1,user+1},3*(j)-1),data(idx{k+1,user+1},3*(j)),data(idx{k+1,user+1},3*j +1),class_colors{j});
        end
    end
    legend('Sensor 1','Sensor 2', 'Sensor 3','Sensor 4');
    xlabel('X');ylabel('Y');zlabel('Z');
    view(40,35);
end

%Now idx contains all of the indexes of the users and exercise labels.
%columns are for each user, rows are for each exercise.

window_size = 50;
data = data(:,2:13);
data_preproc = zeros(size(data));
accel=[];
%% Calculate pitch, roll for each filtered accelerometer reading:
for i = 1:size(data,1)
    for j = 1:4
        accel = data(i,((3*j-2):3*j));
        pitch = atan(accel(2)/sqrt((accel(1)^2)+(accel(3)^2)));
        roll = atan(-accel(1)/accel(3));
        accel_module = norm(accel);
        data_preproc(i,(j*3-2):(j*3)) = [pitch,roll,accel_module];
    end
end

%% Normalize the columns of the data: 
for i = 1:size(data,2)
    data_preproc(:,i) = mat2gray(data_preproc(:,i));
end
% now data preproc contains the normalized pitch, roll, and magnitude of each
% accelerometer, by column: 1)pitch1 2) roll1 3)magnitude1, and so on for
% all four sensors.
%%
%Now split the data up by which exercise and user it was:
data_windowed_nm = zeros(size(data_preproc,1),(size(data_preproc,2) + 24));
data_windowed_nm(:,1:9) = [data_preproc(:,1:3),zeros(size(data_preproc,1),6)];
data_windowed_nm(:,10:18) = [data_preproc(:,4:6),zeros(size(data_preproc,1),6)];
data_windowed_nm(:,19:27) = [data_preproc(:,7:9),zeros(size(data_preproc,1),6)];
data_windowed_nm(:,28:36) = [data_preproc(:,10:12),zeros(size(data_preproc,1),6)];

%prepare an averaging filter of size filter_size:
filter_size = floor(window_size/((window_size)^(1/3)));
f = fspecial('average',[1 filter_size])'; %create an averaging filter
for m = 1:5
    for n = 1:4
        for i = min(idx{m,n}):max(idx{m,n})
            k = max(i - window_size,min(idx{m,n}));
            temp = data_preproc(k:i,:);
            smoothed = imfilter(data_preproc(k:i,:),f);
            
            %Calculate the fourier transforms of the window, take the
            %fundamental: 
            f_fundamental = fft(smoothed,[],1);
            f_fundamental = f_fundamental(1,:);
            %window_mean = mean(smoothed,1);
            window_var = var(smoothed,1);
            if numel(window_var) > 1
                for j = 1:4
                    data_windowed_nm(i,(9*j-5):9*j) = [window_var(3*j-2), window_var(3*j-1), window_var(3*j), f_fundamental((j*3-2):j*3)];
                end
            else
                if (i-1) > 0
                    for j = 1:4
                        data_windowed_nm(i,(9*j-5):9*j) = data_windowed_nm(i-1,(9*j-5):9*j);
                    end
                else
                    for j = 1:4
                        data_windowed_nm(i,(9*j-5):9*j) = zeros(1,6);
                    end
                end
            end
        end
    end
end
%%
%Create scatter plots of the data:
close all; 
b = {'arm','abdomen','thigh','ankle'};
features = {'\theta','\Phi','\alpha','\sigma^2_\theta','\sigma^2_\Phi','\sigma^2_\alpha','F_\theta','F_\Phi','F_\alpha'};
hold on;
for k = 1:4
    figure();
    for j = 1:9
        subplot(3,3,j);
        t = sprintf('Feature: %s    Sensor: %d',features{j},k);
        title(t);
        hold on;
        for i = 1:5
            temp = data_windowed_nm(idx{i,k},9*(k-1)+j);
            h = hist(temp,50)/length(temp);
            h = plot(h,class_colors{i},'LineWidth',4);
        end
        axis tight;
        legend('Sitting','Sitting Down','Standing','Standing Up','Walking');
    end
end

%% Shuffle the data:
idx = randperm(size(data_windowed_nm,1));
data_windowed_nm = data_windowed_nm(idx,:);
target = target(idx,:);
target_bin = zeros(size(target,1),5);
for i = 1:size(data,1)
    if (data(i) == 0)
        target_bin(i,:) = [1 0 0 0 0];
    end
    if (target(i) == 1)
        target_bin(i,:) = [0 1 0 0 0];
    end
    if (target(i) == 2)
        target_bin(i,:) = [0 0 1 0 0];
    end
    if (target(i) == 3)
        target_bin(i,:) = [0 0 0 1 0];
    end
    if (target(i) == 4)
        target_bin(i,:) = [0 0 0 0 1];
    end
end