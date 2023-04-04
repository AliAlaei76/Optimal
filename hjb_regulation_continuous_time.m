clc;
clear;
close all;

%% parameters

mq = 0.3812;
Ixx = 2.661e-5;
Iyy = 2.661e-5;
Izz = 2.661e-5;
k = 2e-6;
L = 0.035;
b = 7e-9;
kd = 0;
rho = 0.037;
r = 0.03;
z_offset = 0.02;
g = 9.8;
ss = diag([1, 1, 0.5]) * 2;

parameters = [mq, Ixx, Iyy, Izz, k, L, b, kd, rho, r, z_offset, g];
xd = 2;
yd = 2;
zd = 1;
%% Matrices

%states = [wx wy wz phi theta psi x y z vx vy vz]

A1 = [0 g 0
    -g 0 0
    0 0 0];

A = [zeros(3, 12);
    eye(3), zeros(3, 9);
    zeros(3, 9), eye(3);
    zeros(3), A1, zeros(3, 6)];

B1 = [L, 0, -L, 0;
    0, L, 0, -L;
    b / k, -b / k, b / k, -b / k];

J = diag([Ixx, Iyy, Izz]);
B = [(J^(-1)) * B1;
    zeros(8, 4);
    ones(1, 4) / mq];

Q = diag([1, 1, 100, 1, 1, 100, 1, 1, 100, 1, 1, 10000]);
R = eye(4);
H = zeros(12);
Rinv = R^ - 1;

%% calculate recutti
Ts = 1e-4;
Tf = 30;
t = 0:Ts:Tf;
N = numel(t);

P = zeros(12, 12, N);
P(:, :, N) = H;

for i = N:-1:2
    P(:, :, i - 1) = P(:, :, i) + Ts * round(Q -P(:, :, i) * B * (Rinv) * B' * P(:, :, i) + P(:, :, i) * A + A' * P(:, :, i) , 5);
end

%%%%

%% HJB

% linear
x = zeros(12, N);
x(:, 1) = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0].';
u = zeros(4, N);

for i = 1:N - 1
    u(:, i) = -R^(-1) * B.' * P(:, :, i) * (x(:, i) - [0, 0, 0, 0, 0, 0, xd, yd, zd, 0, 0, 0].');
    x(:, i + 1) = x(:, i) + Ts * (A * x(:, i) + B * u(:, i));
end

figure
subplot(2, 1, 1);
plot(t(1:round(N)), x([7, 8, 9, 1, 2, 3], 1:round(N)), 'LineWidth', 2);
grid on
grid minor
xlabel('time (sec)');
ylabel('States');
legend('$x$', '$y$', '$z$', '$\phi$', '$\theta$', '$\psi$', 'Interpreter', 'latex')

subplot(2, 1, 2);
plot(t(1:round(N)), sign(u(:, 1:round(N)) + mq * g / 4) .* (abs(u(:, 1:round(N)) + mq * g / 4) / k).^0.5, 'LineWidth', 2);
grid on
grid minor
xlabel('time (sec)');
ylabel('Inputs');
legend('$u_1$', '$u_2$', '$u_3$', '$u_4$', 'Interpreter', 'latex')

% nonlinear
x = zeros(12, N);
x(:, 1) = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0].';
u = zeros(4, N);

for i = 1:N - 1
    u(:, i) = -R^(-1) * B.' * P(:, :, i) * (x(:, i) - [0, 0, 0, 0, 0, 0, xd, yd, zd, 0, 0, 0].');
    m = (u(:, i) + ones(4, 1) * mq * g / 4);
    w1 = sign(m(1)) * (abs(m(1) / k)^0.5);
    w2 = sign(m(2)) * (abs(m(2) / k)^0.5);
    w3 = sign(m(3)) * (abs(m(3) / k)^0.5);
    w4 = sign(m(4)) * (abs(m(4) / k)^0.5);
    [T_ddot, W_dot] = NonlinearDynamics(w1, w2, w3, w4, x(7:9, i), x(10:12, i), x(4:6, i), x(1:3, i), parameters);
    x(:, i + 1) = x(:, i) + Ts * ([W_dot; W2E(x(1:3, i), x(4:6, i)); x(10:12, i); T_ddot]);
end

figure
subplot(2, 1, 1);
plot(t(1:round(N)), x([7, 8, 9, 1, 2, 3], 1:round(N)), 'LineWidth', 2);
grid on
grid minor
xlabel('time (sec)');
ylabel('States');
legend('$x$', '$y$', '$z$', '$\phi$', '$\theta$', '$\psi$', 'Interpreter', 'latex')

subplot(2, 1, 2);
plot(t(1:round(N)), sign(u(:, 1:round(N)) + mq * g / 4) .* (abs(u(:, 1:round(N)) + mq * g / 4) / k).^0.5, 'LineWidth', 2);
grid on
grid minor
xlabel('time (sec)');
ylabel('Inputs');
legend('$u_1$', '$u_2$', '$u_3$', '$u_4$', 'Interpreter', 'latex')

figure
plot(t(1:round(N)), reshape(P(:, :, 1:round(N)), 144, round(N)), 'LineWidth', 2);
grid on
grid minor
xlabel('time (sec)');
ylabel('$P$', 'interpreter', 'latex');

%% functions

function [T_ddot, W_dot] = NonlinearDynamics(w1, w2, w3, w4, T, T_dot, Theta, W, parameters)

    %% parameters

    mq = parameters(1);
    Ixx = parameters(2);
    Iyy = parameters(3);
    Izz = parameters(4);
    k = parameters(5);
    L = parameters(6);
    b = parameters(7);
    kd = parameters(8);
    rho = parameters(9);
    r = parameters(10);
    z_offset = parameters(11);
    g = parameters(12);

    %% states

    x = T(1);
    y = T(2);
    z = T(3);

    phi = Theta(1);
    theta = Theta(2);
    psi = Theta(3);

    %% Matrices

    Rx = [1 0 0; 0 cos(phi) -sin(phi); 0 sin(phi) cos(phi)];
    Ry = [cos(theta) 0 sin(theta); 0 1 0; -sin(theta) 0 cos(theta)];
    Rz = [cos(psi) -sin(psi) 0; sin(psi) cos(psi) 0; 0 0 1];
    R = Rz * Ry * Rx;

    I = diag([Ixx, Iyy, Izz]);

    %% Input Modification

    in = zeros(4, 1);
    Ground_Effect_Coef = @(z) 1;
    z1 = [0 0 1] * ([x; y; z] + R * [0; L; z_offset]);
    z2 = [0 0 1] * ([x; y; z] + R * [-L; 0; z_offset]);
    z3 = [0 0 1] * ([x; y; z] + R * [0; -L; z_offset]);
    z4 = [0 0 1] * ([x; y; z] + R * [L; 0; z_offset]);
    in(1) = Ground_Effect_Coef(z1) * w1 * abs(w1);
    in(2) = Ground_Effect_Coef(z2) * w2 * abs(w2);
    in(3) = Ground_Effect_Coef(z3) * w3 * abs(w3);
    in(4) = Ground_Effect_Coef(z4) * w4 * abs(w4);

    %% Translational Dynamics

    TB = k * [0; 0; sum(in)]; %trust force with ground effect
    FD = -kd * T_dot; %frictional force
    T_ddot = [0; 0; -g] + (R * TB + FD) / mq; %Translational Dynamics

    %% Rotational Dynamics

    tau = [L * k * (in(1) - in(3)); L * k * (in(2) - in(4)); b * (in(1) - in(2) + in(3) - in(4))]; %Rotational Torque with ground effect
    W_dot = I \ (tau - cross(W, I * W)); %Rotational Dynamics

end

function Theta_dot = W2E(W, Theta)

    phi = Theta(1);
    theta = Theta(2);
    psi = Theta(3);

    E = [1, sin(phi) * tan(theta), cos(phi) * tan(theta);
        0, cos(phi), -sin(phi);
        0, sin(phi) / cos(theta), cos(phi) / cos(theta); ];

    Theta_dot = E * W;
end
