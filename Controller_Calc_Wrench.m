close all
clear
clc

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
B = [J^ - 1 * B1;
    zeros(8, 4);
    ones(1, 4) / mq];

Q = diag([1, 1, 100, 1, 1, 100, 1, 1, 100, 1, 1, 10000]);
R = eye(4);

[K, S, E] = lqr(A, B, Q, R);

%open('quad_simulation_with_linearization_around_wrench.slx');
sim('quad_simulation_with_linearization_around_wrench.slx');

figure
subplot(2, 1, 1);
plot(Positions.Time, Positions.Data, 'LineWidth', 2);
hold on
plot(Angles.Time, Angles.Data, 'LineWidth', 2);
grid on
grid minor
xlabel('time (sec)');
ylabel('States');
legend('$x$', '$y$', '$z$', '$\phi$', '$\theta$', '$\psi$', 'Interpreter', 'latex')

subplot(2, 1, 2);
plot(Inputs.Time, Inputs.Data, 'LineWidth', 2);
grid on
grid minor
xlabel('time (sec)');
ylabel('Inputs');
legend('$u_1$', '$u_2$', '$u_3$', '$u_4$', 'Interpreter', 'latex')

%% simulation

Ts = 1e-4;
Tf = 10;
t = 0:Ts:Tf;
N = numel(t);

x = zeros(12, N);
u = zeros(4, N);
x(:, 1) = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0].';

for i = 1:N - 1
    u(:, i) = -K * (x(:, i) - [0, 0, 0, 0, 0, 0, xd, yd, zd, 0, 0, 0].');
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
