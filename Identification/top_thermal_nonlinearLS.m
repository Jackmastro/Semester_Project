clc; close all; clear;

%% Load data
load("HP_fitted_data.mat");

%% Compute values
N = 
x12 = [x12_1, x12_2, ..., x12_N];
x14 = [x14_1, x14_2, ..., x14_N];
x22 = [x22_1, x22_2, ..., x22_N];
x23 = [x23_1, x23_2, ..., x23_N];
x3 = [x3_1, x3_2, ..., x3_N];
y1 = [y1_1, y1_2, ..., y1_N];
y3 = [y3_1, y3_2, ..., y3_N];

%% Initial guesses for [theta1, theta2, theta3, theta4]
initial_guess = [10; 1/5; 1/25; (273+39)];

% Call lsqnonlin to minimize the sum of squared residuals
options = optimoptions('lsqnonlin', 'Display', 'iter');
[theta_estimates, resnorm] = lsqnonlin(@myResiduals, initial_guess, [], [], options);

% Display results
disp('Residual norm:')
disp(resnorm)
disp('Estimated parameters:')
fprintf('R_3^{lamdba} = %.2f\n', 1/theta_estimates(1));
fprintf('R_Al^{lamdba} = %.2f\n', 1/theta_estimates(2));
fprintf('R_i = %.2f\n', 1/theta_estimates(3));
fprintf('T_c = %.2f\n', theta_estimates(4));

%% Define the function that calculates residuals
function residuals = myResiduals(theta)
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
        R2 = x22(i) * theta2 + x23(i) * theta3;
        R3 = (theta4 - x3(i)) * theta1 + x14(i) * theta4 - y3(i);

        % Append the residuals to the overall residual vector
        residuals(i*3-2:i*3) = [R1; R2; R3];
    end
end

