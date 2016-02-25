%% Naive Bayes Classification:

PreProcessing %generate data
%Split the data into testing and training:
p = .8;
train_length = floor(p*length(target));
data_train = data_windowed_nm(1:train_length,:);
data_test = data_windowed_nm((train_length):length(target),:);
target_train = target(1:train_length);
target_test = target((train_length):length(target));

% %% Naive Bayes Classifier:
NBModel = fitcnb(data_train,target_train);
[label] = predict(NBModel,data_test);
C = confusionmat(target_test,label);

% Calculate the misclassification percentage for activities 1-5: 
e = [C(1,1)/sum(C(1,:)), ...
        C(2,2)/sum(C(2,:)), ...
        C(3,3)/sum(C(3,:)), ...
        C(4,4)/sum(C(4,:)), ...
        C(5,5)/sum(C(5,:))];
% e1 = [0.9846    0.6770    0.9010    0.4868    0.8250]
% e2 = [0.9842    0.6824    0.9039    0.4966    0.8275]
% e3 = [0.9843    0.6891    0.9040    0.4733    0.8315]
% e4 = [0.9865    0.6953    0.9054    0.4581    0.8319]
% e5 = [0.9866    0.6817    0.9036    0.4871    0.8294]
% e6 = [0.9834    0.6664    0.9023    0.4906    0.8353]
% e7 = [0.9840    0.6739    0.9028    0.4758    0.8274]
% e8 = [0.9842    0.6993    0.9013    0.4752    0.8272]
% e9 = [0.9849    0.6931    0.9023    0.4769    0.8329]
% e10 = [0.9841    0.6744    0.9006    0.4890    0.8271]