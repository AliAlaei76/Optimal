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

%states = [wx wy wz phi theta psi x y z vx vy vz]

Q = diag([1, 1, 100, 1, 1, 100, 1, 1, 100, 1, 1, 10000]);
R = eye(4);
H = zeros(12);

[K, S, ~] = lqr(A, B, Q, R);
H = S;
%% Steepest Descent


xs = [0, 0, 0, 0, 0, 0, xd, yd, zd, 0, 0, 0].';
Hx = @(x, u, p) Q * (x - xs) + fx(x, u, parameters).' * p;
Hu = @(x, u, p) R * u + fu(x, u, parameters).' * p;

alpha = 1e-7;

Ts = 1e-5;
Tf = 30;
t = 0:Ts:Tf;
N = numel(t);
x = zeros(12, N);
u = zeros(4, N);
p = zeros(12, N);

x(:, 1) = [0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0];

for i = 1:N - 1
    u(:, i) = -K * (x(:, i) - xs);
    x(:, i + 1) = x(:, i) + Ts * f(x(:, i), u(:, i), parameters);
    p(:, i) = S * (x(:, i) - xs);
end

u(:, N) = -K * (x(:, N) - xs);
p(:, N) = S * (x(:, N) - xs);

figure
subplot(2, 1, 1);
plt1 = plot(t, x([7, 8, 9, 1, 2, 3], :), 'LineWidth', 2);
Error = 0;
for i = 1:N
    u(:, i) = u(:, i) - alpha * Hu(x(:, i), u(:, i), p(:, i));
    Error = Error + norm(alpha * Hu(x(:, i), u(:, i), p(:, i)));
end
it = 0;
plt1_T = title(sprintf('step : %d , Error : %0.5e', it, Error));
AAA = [7, 8, 9, 1, 2, 3];
it = it + 1;
grid on
grid minor
xlabel('time (sec)');
ylabel('States');
legend('$x$', '$y$', '$z$', '$\phi$', '$\theta$', '$\psi$', 'Interpreter', 'latex')

subplot(2, 1, 2);
plt2 = plot(t(1:round(N)), sign(u(:, 1:round(N)) + mq * g / 4) .* (abs(u(:, 1:round(N)) + mq * g / 4) / k).^0.5, 'LineWidth', 2);
%plt2_T = title(sprintf('step : %d , Error : %0.5e', 0, Error));
grid on
grid minor
xlabel('time (sec)');
ylabel('Inputs');
legend('$u_1$', '$u_2$', '$u_3$', '$u_4$', 'Interpreter', 'latex')
ylim([620 720]);


%{
for Z = 1:10

    x(:, 1) = [0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0];

    for i = 1:N - 1
        x(:, i + 1) = x(:, i) + Ts * f(x(:, i), u(:, i), parameters);
    end

    p(:, N) = S * (x(:, N) - xs);

    for i = N:-1:2
        p(:, i - 1) = p(:, i) + Ts * Hx(x(:, i), u(:, i), p(:, i));
    end

    Error = 0;

    for i = 1:N
        u(:, i) = u(:, i) - alpha * Hu(x(:, i), u(:, i), p(:, i));
        Error = Error + norm(alpha * Hu(x(:, i), u(:, i), p(:, i)));
    end

    for i = 1:6
        plt1(i).YData = x(AAA(i), :);
    end

    for i = 1:size(u, 1)
        plt2(i).YData = sign(u(i, :) + mq * g / 4) .* (abs(u(i, :) + mq * g / 4) / k).^0.5;
    end

    plt1_T.String = sprintf('step : %d , Error : %0.5e', it, Error);
    it = it + 1;
    drawnow
end

%}

%% functions

function xdot = f(x, u, parameters)
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
    m = (u + ones(4, 1) * mq * g / 4);
    w1 = sign(m(1)) * (abs(m(1) / k)^0.5);
    w2 = sign(m(2)) * (abs(m(2) / k)^0.5);
    w3 = sign(m(3)) * (abs(m(3) / k)^0.5);
    w4 = sign(m(4)) * (abs(m(4) / k)^0.5);
    [T_ddot, W_dot] = NonlinearDynamics(w1, w2, w3, w4, x(7:9), x(10:12), x(4:6), x(1:3), parameters);
    xdot = [W_dot; W2E(x(1:3), x(4:6)); x(10:12); T_ddot];
end

function y = fx(x, u, parameters)
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
    m = (u + ones(4, 1) * mq * g / 4);

    wx = x(1);
    wy = x(2);
    wz = x(3);
    phi = x(4);
    theta = x(5);
    psi = x(6);

    W_dot_X = [
            [0, Izz * wz - Iyy * wz, Izz * wy - Iyy * wy];
            [Ixx * wz - Izz * wz, 0, Ixx * wx - Izz * wx];
            [Iyy * wy - Ixx * wy, Iyy * wx - Ixx * wx, 0];
            ];

    E = [1, sin(phi) * tan(theta), cos(phi) * tan(theta);
        0, cos(phi), -sin(phi);
        0, sin(phi) / cos(theta), cos(phi) / cos(theta);
        ];

    E1 = [
        [wy * cos(phi) * tan(theta) - wz * sin(phi) * tan(theta), wz * cos(phi) * (tan(theta)^2 + 1) + wy * sin(phi) * (tan(theta)^2 + 1), 0];
        [- wz * cos(phi) - wy * sin(phi), 0, 0];
        [(wy * cos(phi)) / cos(theta) - (wz * sin(phi)) / cos(theta), (wz * cos(phi) * sin(theta)) / cos(theta)^2 + (wy * sin(phi) * sin(theta)) / cos(theta)^2, 0];
        ];

    S = [
        [cos(phi) * sin(psi) - cos(psi) * sin(phi) * sin(theta), cos(phi) * cos(psi) * cos(theta), cos(psi) * sin(phi) - cos(phi) * sin(psi) * sin(theta)];
        [- cos(phi) * cos(psi) - sin(phi) * sin(psi) * sin(theta), cos(phi) * cos(theta) * sin(psi), sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta)];
        [-cos(theta) * sin(phi), -cos(phi) * sin(theta), 0];
        ];

    y = [W_dot_X, zeros(3, 9); E, E1, zeros(3, 6); zeros(3, 9), eye(3); zeros(3), S * sum(m), zeros(3), -kd * eye(3)];
end

function y = fu(x, u, parameters)
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
    m = (u + ones(4, 1) * mq * g / 4);

    wx = x(1);
    wy = x(2);
    wz = x(3);
    phi = x(4);
    theta = x(5);
    psi = x(6);

    TT = [L * k, 0, -L * k, 0;
        0, L * k, 0, -L * k;
        b, -b, b, -b];

    Rx = [1 0 0; 0 cos(phi) -sin(phi); 0 sin(phi) cos(phi)];
    Ry = [cos(theta) 0 sin(theta); 0 1 0; -sin(theta) 0 cos(theta)];
    Rz = [cos(psi) -sin(psi) 0; sin(psi) cos(psi) 0; 0 0 1];
    R = Rz * Ry * Rx;

    y = [TT; zeros(3, 4); zeros(3, 4); k * R * [0, 0, 0, 0; 0, 0, 0, 0; 1, 1, 1, 1]];
end

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
