%% parameters
    
% trajectory file
% beacons: matrix of dimensions n_beacons x 2 [x_1 y_1 ; ... ; x_n y_n]
% radius: operative area of the beacons
% motion_model: 'P' or 'PV'
% measurement_model: 'euclidean' or 'rssi'
% sampling_time: how often we take the measurements from the beacons
% var_z: error variance of the beacons        
%     

function [prediction, dist_err, dist_max,  RMSE_x, RMSE_y, RMSE_net] = ekf_tracking(name_trajectory, ...
                                                                        beacons, ...
                                                                        radius, ...
                                                                        motion_model, ...
                                                                        measurement_model, ...
                                                                        sampling_time, ...
                                                                        var_z...
                                                                       )
%% load the trajectory
file = load(name_trajectory);
X = file.X;

N = size(X,1);

% define polling query of sensors
dt = sampling_time;
% number of sensors
p = size(beacons,1);
% dimension of the states : 2D motion x,y
n = 2;


%% plot the trajectory and sensors position
figure;
plot(X(:,1),X(:,2));
hold on;
plot(beacons(:,1),beacons(:,2), 'x');
t = linspace(0,2*pi);
for j=1:size(beacons,1)
    plot(radius*cos(t)+beacons(j,1),radius*sin(t)+beacons(j,2),'--');
end


if (strcmp(motion_model,'P'))
    %% define the matrix of motion equation
    F = [1,0 ; 0,1]; % brownian motion

    %% process covariance
    Ex = eye(n);

    %% output covariance : each sensor has its own uncertainty and it's uncorrelated with the others
    Ez = eye(p);
    
    %% Initialization
    x_hat = X(1,:)';
    
elseif (strcmp(motion_model,'PV'))
  
%     count = 0;
%     for t=dt+1:dt:N
%         v(t) =  sqrt((X(t,1)-X(t-dt,1))^2 + (X(t,2)-X(t-dt,2))^2 )/dt;
%         delta_v(t) = abs(v(t)-v(t-dt));
%         count = count+1;
%     end
% 
%     v_mean = sum(v)/count;
%     delta_v_mean = sum(delta_v)/count;

    %% process covariance : velocity acceleration model
    % accel_noise_mag = .001; % process noise: the variability in how fast the target is speeding up (stdv of acceleration: meters/sec^2)
    % accel_noise_mag =(delta_v_mean)^2 / dt; % NOT WORKING AS EXPECTED PROBABLY BECAUSE OF DATASET
    accel_noise_mag =(0.001)^2 / dt; %% tuned by hand
    
    % Ex = [  dt^4/4 0 dt^3/2 0; ...
    %         0 dt^4/4 0 dt^3/2; ...
    %         dt^3/2 0 dt^2 0; ...
    %         0 dt^3/2 0 dt^2] .* accel_noise_mag^2; % Ex convert the process noise (stdv) into covariance matrix
    Ex = [  dt^3/3 0 dt^2/2 0; ...
            0 dt^3/3 0 dt^2/2; ...
            dt^2/2 0 dt 0; ...
            0 dt^2/2 0 dt] .* accel_noise_mag; % Ex convert the process noise (stdv) into covariance matrix

    %% [state transition (state + velocity)] + [input control (acceleration)]
    F = [1 0 dt 0; ...
         0 1 0 dt; ...
         0 0 1 0; ...
         0 0 0 1]; %state update matrice
    
    %% Initialization
    x_hat = [X(1,1); X(1,2); 0; 0 ];
    
    
    
    %% output covariance : each sensor has its own uncertainty and it's uncorrelated with the others
    
    Ez = eye(p)*var_z;
    
end

P = Ex; % estimate of initial position variance (covariance matrix)
prediction = x_hat';

if (strcmp(measurement_model, 'euclidean'))    
    noised_distances = zeros(size(beacons,1),N);
elseif (strcmp(measurement_model, 'rssi'))
    %% parameters of the measurement model (taken from Peerapong et.Al)
    Pd_0 = 3.0; % RSSI value at 1m [dBm]
    R_0  = 1.0; % reference or breakpoint distance [m] 
    l = 3.0;    % path loss exponent
    
    radio_power = zeros(size(beacons,1),N);
    noised_radio_power = zeros(size(beacons,1),N);
end


%% Kalman filter
distances = zeros(size(beacons,1),N);
for t=1:N
     for k=1:size(beacons,1)
         
         distances(k,t) = sqrt((X(t,1)-beacons(k,1)).^2 + (X(t,2)-beacons(k,2)).^2);
         
         if (strcmp(measurement_model, 'rssi'))
            radio_power(k,t) = Pd_0 - 5*l*log10(distances(k,t));
         end
         
         if distances(k,t) > radius
             distances(k,t) = 0;
         else
             if (strcmp(measurement_model, 'rssi'))
                noised_radio_power(k,t) = radio_power(k,t) + sqrt(var_z)*randn(1);
             elseif (strcmp(measurement_model, 'euclidean'))
                noised_distances(k,t) = distances(k,t) + sqrt(var_z)*randn(1);
             end
             
         end
         
     end
end

number_est = 1;
dist_max = 0;
for t=1:dt:N

    if (strcmp(measurement_model, 'euclidean'))
        z = noised_distances(:,t);
    elseif (strcmp(measurement_model, 'rssi'))
        z = noised_radio_power(:,t);
    end
    
    h = zeros(size(beacons,1),1);
    for k=1:size(beacons,1)
        if z(k) ~=0
            h(k) = sqrt((x_hat(1) - beacons(k,1)).^2 + (x_hat(2) - beacons(k,2)).^2);
            if (strcmp(measurement_model, 'rssi'))
                h(k) = Pd_0 - 5*l*log10(h(k));
            end
        end
    end
    
    %% building H matrix
    H = [];
    active_sensors = 0;
    for i=1:size(z,1)
        if z(i) ~= 0 % si linearizza e si calcola nella predizione precedente
            
            if (strcmp(measurement_model, 'euclidean'))
                dh_dx = (x_hat(1)-beacons(i,1))/sqrt((x_hat(1)-beacons(i,1))^2 + (x_hat(2)-beacons(i,2))^2);
                dh_dy = (x_hat(2)-beacons(i,2))/sqrt((x_hat(1)-beacons(i,1))^2 + (x_hat(2)-beacons(i,2))^2);
            elseif (strcmp(measurement_model, 'rssi'))
                dh_dx = -5*l*2*(x_hat(1)-beacons(i,1)) / (log(10)*((x_hat(1)-beacons(i,1))^2 + (x_hat(2)-beacons(i,2))^2));
                dh_dy = -5*l*2*(x_hat(2)-beacons(i,2)) / (log(10)*((x_hat(1)-beacons(i,1))^2 + (x_hat(2)-beacons(i,2))^2));
            end
            active_sensors = active_sensors + 1;
            
            if (strcmp(motion_model, 'P'))
                H = [H;  dh_dx dh_dy];
            elseif (strcmp(motion_model, 'PV'))
                H = [H;  dh_dx dh_dy 0 0 ];
            end
            
        else % in questo caso non si ha la misurazione del beacon perchè troppo lontano dal target -> riga nulla
            if (strcmp(motion_model, 'P'))
                H = [H; 0 0];
            elseif (strcmp(motion_model, 'PV'))
                H = [H; 0 0 0 0];
            end
            
        end
    end
    if active_sensors<2
        disp('Attenzione sensori rilevati inferiori a 2');
    end
    
    
    %% prediction step
    
    % project the state ahead
    x_hat = F * x_hat;
    % project the covariance ahead
    P_hat = F*P*F' + Ex;
    
    %% correction step

    % compute the kalman gain
    K = P_hat*H' * inv(H*P_hat*H' + Ez); 
    
    % update the state estimate with measurement z(t)
    x_hat = x_hat + K*(z-h);
    % update the error covariance
    
    if (strcmp(motion_model, 'P'))
        P = (eye(2)-K*H)*P_hat;
    elseif (strcmp(motion_model, 'PV'))
        P = (eye(4)-K*H)*P_hat;
    end
    
    prediction = [prediction; x_hat'];
    
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

dist_err = sum(dist) / number_est;
RMSE_x = sqrt(sum(x_err.^2)/number_est);
RMSE_y = sqrt(sum(y_err.^2)/number_est);
RMSE_net = sqrt(RMSE_x.^2 + RMSE_y.^2);