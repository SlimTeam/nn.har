%%
PreProcessing
x = data_windowed_nm';
t = target_bin';
net = patternnet([75,50]);
view(net)
%%
[net,tr] = train(net,x,t);
view(net)
y = net(x);
perf = perform(net,t,y);
classes = vec2ind(y);
perf

%Cross entropy performances from 15 different tests stored in results:
% results = [2.67e-4, 1.158e-4, 3.9697e-4, 6.1987e-4, 5.8717e-4, 5.566e-4, ...
%             5.3401e-4, 6.4229e-4, 4.1692e-4, 9.3177e-4, 3.9054e-4, ...
%             7.1764e-4, 3.6716e-4, 8.1447e-4, 4.8601e-4];
% mean_ce = mean(results);
% std_ce = sqrt(var(results));
