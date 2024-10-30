clc; close all; clear;

%% Measured data
U_sys = 4.096; % V
T_amb = convtemp(23.4, 'C', 'K'); % K
Q_LED = 7.7/1000 * U_sys * 24; % W

%% Load data
load("HP_fitted_data.mat");
data_0 = readtable("20241030_162031_diya06.csv");

%% Compute values
T_LED = convtemp(mean([data_0.T1_C, data_0.T2_C], 2), 'C', 'K'); % K, mean along the rows
T_cell = T_LED - 1; % K
I_HP = max(zeros(size(data_0.I_HP_mA)), data_0.I_HP_mA ./ 1000); % only positive values
% I_HP = data_0.I_HP_mA ./ 1000; % only positive values
x_HP = data_0.x_HP;

N = height(data_0);

x12 = T_LED - T_cell;
x14 = S_M * I_HP;
x23 = T_amb - T_cell;
x314 = T_LED;
y1 = 0.5 * R_M * I_HP.^2 + K_M * (U_sys * x_HP - R_M * I_HP) / S_M + Q_LED;
y3 = y1 - Q_LED;

%% Initial guesses for [theta1, theta2, theta3, theta4]
initial_guess = [10; 1/5; 1/25; (273+39)];

options = optimoptions('lsqnonlin', 'Display', 'iter');

lower_bound = [0; 0; 0; 0];
upper_bound = [500; 10; 2; 330];

% Call lsqnonlin with bounds
[theta_estimates, resnorm] = lsqnonlin(@(theta) myResiduals(theta, N, x12, x14, x23, x314, y1, y3), initial_guess, lower_bound, upper_bound, options);


% Display results
disp('Residual norm:')
disp(resnorm)
disp('Estimated parameters:')
fprintf('R_3^{lamdba} = %.2f\n', 1/theta_estimates(1));
fprintf('R_Al^{lamdba} = %.2f\n', 1/theta_estimates(2));
fprintf('R_i = %.2f\n', 1/theta_estimates(3));
fprintf('T_c = %.2f\n', theta_estimates(4));

%% Define the function that calculates residuals
function residuals = myResiduals(theta, N, x12, x14, x23, x314, y1, y3)
    % Extract parameters
    theta1 = theta(1);
    theta2 = theta(2);
    theta3 = theta(3);
    theta4 = theta(4);

    % Initialize residuals
    residuals = NaN(N*3,1);

    % Loop over each measurement
    for i = 1:N
        % Define the residuals for each measurement
        R1 = x12(i) * theta2 + x14(i) * theta4 - y1(i);
        R2 = x12(i) * theta2 + x23(i) * theta3;
        R3 = (theta4 - x314(i)) * theta1 + x14(i) * theta4 - y3(i);

        % Append the residuals to the overall residual vector
        residuals(i*3-2:i*3) = [R1; R2; R3];
    end
end

