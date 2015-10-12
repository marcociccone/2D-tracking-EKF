%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Trilateration approach with euclidean distance measurements%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;
%% load the trajectory
file = load('hard2.mat');
%file = load('easy2.mat');
X = file.X;
%% normalization for a room of dimensions 30x10 m

N = size(X,1);
%% define the positions of the sensors
radius = 10;% this is how far the sensor can work 
s = [%7.5,3.5;   ...
     15,5;      ...
     %20,3.5;    ...
     %20,7.5;    ...
     7.5,7.5;   ...
     %10,10;     ...
     10,2;      ...
     %17.5,10;   ...
     %17.5,2;    ...
    ];

% define polling query of sensors
dt = 10;
% number of sensors
p = size(s,1);
% dimension of the states : 2D motion x,y
n = 2;


%% plot the trajectory and sensors position
figure;
plot(X(:,1),X(:,2));
hold on;
plot(s(:,1),s(:,2), 'x');
t = linspace(0,2*pi);
for j=1:size(s,1)
    plot(radius*cos(t)+s(j,1),radius*sin(t)+s(j,2),'--');
end



%% Initialization
x_hat = [X(1,:)'];% ;0 ;0];
prediction = x_hat';

distances = zeros(size(s,1),N);
noised_distances = zeros(size(s,1),N);

number_est = 1;
dist_max = 0;
predicted = [];
for t=1:N
     for k=1:size(s,1)
         distances(k,t) = sqrt((X(t,1)-s(k,1)).^2 + (X(t,2)-s(k,2)).^2);
          %if distances(k,t) > radius
          %   distances(k,t) = NaN;
          %else
             noised_distances(k,t) = awgn(distances(k,t),10);
          %end         
     end
    
    %% TRILATERATION WITH LEAST SQUARES
    
    A = [];
    b = [];
    for i=1:p-1    % build matrix A and b
        A = [A ; s(p,1)-s(i,1) s(p,2)-s(i,2)];
        %b = [b ; 0.5*((distances(i,t).^2 - distances(p,t).^2) - (s(i,1).^2 - s(p,1).^2) - (s(i,2).^2 - s(p,2).^2))];
        b = [b ; 0.5*((noised_distances(i,t).^2 - noised_distances(p,t).^2) - (s(i,1).^2 - s(p,1).^2) - (s(i,2).^2 - s(p,2).^2))];
    end
    x_hat = inv(A'*A)*A'*b; % compute the trilateration with least squares
    predicted = [predicted ; x_hat'];
    
    %% compute distance error 
    
    x_err(number_est) = X(t,1) - x_hat(1);
    y_err(number_est) = X(t,2) - x_hat(2);
    
    dist(number_est) = sqrt(x_err(number_est).^2 + y_err(number_est).^2);
    if dist(number_est) > dist_max
        dist_max = dist(number_est);
        pos_dist_max = number_est;
    end
    
    number_est = number_est+1;
    
end

trilat_prediction = predicted(1:dt:N,:);
plot(trilat_prediction(:,1),trilat_prediction(:,2),'Color','Green');

dist_err = sum(dist) / number_est;
RMSE_x = sqrt(sum(x_err.^2)/number_est);
RMSE_y = sqrt(sum(y_err.^2)/number_est);
RMSE_net = sqrt(RMSE_x.^2 + RMSE_y.^2);

disp(['Distance Error  Avg: ',num2str(dist_err)]);
disp(['Distance Error Max : ',num2str(dist_max)]);
disp(['RMSE_x : ',num2str(RMSE_x)]);
disp(['RMSE_y : ',num2str(RMSE_y)]);
disp(['RMSE_net : ',num2str(RMSE_net)]);
