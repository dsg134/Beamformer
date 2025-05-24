% -----------------------------------------------
% Input Parameters
f = 3e9;              % Frequency in Hz
theta_desired = -30;  % Elevation angle in degrees
phi_desired = 30;     % Azimuth angle in degrees
element_spacing = 0.5; % Element spacing in wavelengths (default 0.5 for λ/2)
% -----------------------------------------------

c = 3e8;  % Speed of light (m/s)
lambda = c / f;  % Wavelength
d = element_spacing * lambda;  % Actual element spacing in meters
N = 4; M = 4;    % Array dimensions

% Angle grid for plotting
theta = linspace(-90, 90, 180);  % Elevation
phi = linspace(-90, 90, 180);    % Azimuth
[THETA, PHI] = meshgrid(theta, phi);
THETA_rad = deg2rad(THETA);
PHI_rad = deg2rad(PHI);
theta_desired_inv = -theta_desired; % Adjust as per original code

% Desired steering direction in radians
theta0 = deg2rad(theta_desired_inv);
phi0 = deg2rad(phi_desired);

% Create interference steering vectors for all three complementary directions
theta_int1 = deg2rad(-theta_desired_inv);  % (-θ, +φ)
phi_int1 = deg2rad(phi_desired);

theta_int2 = deg2rad(theta_desired_inv);   % (+θ, +φ)
phi_int2 = deg2rad(-phi_desired);

theta_int3 = deg2rad(-theta_desired_inv);  % (-θ, -φ)
phi_int3 = deg2rad(-phi_desired);

% Steering vector for desired direction
steer = zeros(N, M);
for n = 0:N-1
    for m = 0:M-1
        steer(n+1, m+1) = exp(1j * 2 * pi * d / lambda * ...
            (n * sin(theta0) * cos(phi0) + m * sin(theta0) * sin(phi0)));
    end
end
steer_vec = steer(:) / norm(steer(:));  % Vectorize and normalize

% Steering vectors for interference directions
steer_int1 = zeros(N, M);
steer_int2 = zeros(N, M);
steer_int3 = zeros(N, M);

for n = 0:N-1
    for m = 0:M-1
        % Interference 1: (-θ, +φ)
        steer_int1(n+1, m+1) = exp(1j * 2 * pi * d / lambda * ...
            (n * sin(theta_int1) * cos(phi_int1) + m * sin(theta_int1) * sin(phi_int1)));
        
        % Interference 2: (+θ, -φ)
        steer_int2(n+1, m+1) = exp(1j * 2 * pi * d / lambda * ...
            (n * sin(theta_int2) * cos(phi_int2) + m * sin(theta_int2) * sin(phi_int2)));
        
        % Interference 3: (-θ, -φ)
        steer_int3(n+1, m+1) = exp(1j * 2 * pi * d / lambda * ...
            (n * sin(theta_int3) * cos(phi_int3) + m * sin(theta_int3) * sin(phi_int3)));
    end
end

steer_int1_vec = steer_int1(:) / norm(steer_int1(:));
steer_int2_vec = steer_int2(:) / norm(steer_int2(:));
steer_int3_vec = steer_int3(:) / norm(steer_int3(:));

% Compute Array Factor for Conventional Beamformer
AF_conv = zeros(size(THETA_rad));
for n = 0:N-1
    for m = 0:M-1
        AF_conv = AF_conv + steer(n+1, m+1) * ...
            exp(1j * 2 * pi * d / lambda * ...
            (n * sin(THETA_rad) .* cos(PHI_rad) + ...
             m * sin(THETA_rad) .* sin(PHI_rad)));
    end
end
AF_conv_mag = abs(AF_conv);
AF_conv_db = 20 * log10(AF_conv_mag / max(AF_conv_mag(:)));

% MVDR Beamformer with nulls at all three complementary directions
sigma_n = 0.01; % Noise power
sigma_i = 10;   % Interference power

% Construct covariance matrix with all three interferences
R = sigma_n * eye(N*M) + ...
    sigma_i * (steer_int1_vec * steer_int1_vec') + ...
    sigma_i * (steer_int2_vec * steer_int2_vec') + ...
    sigma_i * (steer_int3_vec * steer_int3_vec');

R_inv = inv(R);
w_mvdr = R_inv * steer_vec / (steer_vec' * R_inv * steer_vec); % MVDR weights
w_mvdr = reshape(w_mvdr, [N, M]); % Reshape to array form

% Compute Array Factor for MVDR Beamformer
AF_mvdr = zeros(size(THETA_rad));
for n = 0:N-1
    for m = 0:M-1
        AF_mvdr = AF_mvdr + w_mvdr(n+1, m+1) * ...
            exp(1j * 2 * pi * d / lambda * ...
            (n * sin(THETA_rad) .* cos(PHI_rad) + ...
             m * sin(THETA_rad) .* sin(PHI_rad)));
    end
end
AF_mvdr_mag = abs(AF_mvdr);
AF_mvdr_db = 20 * log10(AF_mvdr_mag / max(AF_mvdr_mag(:)));

% Window Function Implementations

% 1. Taylor Window Implementation
function w = taylorwin(n, nbar, sll)
    % n = number of points
    % nbar = number of nearly constant-level sidelobes
    % sll = desired sidelobe level in dB (negative value)
    
    A = acosh(10^(-sll/20))/pi;
    w = ones(n, 1);
    
    for m = 1:nbar-1
        w_m = 1;
        for p = 1:(nbar-1)
            if p ~= m
                w_m = w_m * (1 - (m^2/(A^2 + (p-0.5)^2)) / (1 - m^2/p^2));
            end
        end
        for k = 0:n-1
            w(k+1) = w(k+1) + 2 * w_m * cos(2*pi*m*(k - (n-1)/2)/n);
        end
    end
    
    % Normalize
    w = w / max(w);
end

% 2. Chebyshev Window Implementation
function w = chebwin(n, sll)
    % n = number of points
    % sll = sidelobe level in dB (negative value)
    
    beta = cosh(acosh(10^(-sll/20))/(n-1));
    k = 0:n-1;
    x = beta * cos(pi*k/(n-1));
    
    % Chebyshev polynomial
    T = zeros(size(x));
    T(abs(x) <= 1) = cos((n-1)*acos(x(abs(x) <= 1)));
    T(abs(x) > 1) = cosh((n-1)*acosh(abs(x(abs(x) > 1))));
    
    w = T / max(T);
end

% 3. Hamming Window Implementation
function w = hammingwin(n)
    % n = number of points
    w = 0.54 - 0.46 * cos(2*pi*(0:n-1)'/(n-1));
end

% 4. Hann Window Implementation
function w = hannwin(n)
    % n = number of points
    w = 0.5 * (1 - cos(2*pi*(0:n-1)'/(n-1)));
end

% Create 2D windows using outer product
taylor_win_1D = taylorwin(N, 5, -30) * taylorwin(M, 5, -30)';
cheb_win_1D = chebwin(N, -30) * chebwin(M, -30)';
hamming_win_1D = hammingwin(N) * hammingwin(M)';
hann_win_1D = hannwin(N) * hannwin(M)';

% Normalize windows
taylor_win_2D = taylor_win_1D / max(taylor_win_1D(:));
cheb_win_2D = cheb_win_1D / max(cheb_win_1D(:));
hamming_win_2D = hamming_win_1D / max(hamming_win_1D(:));
hann_win_2D = hann_win_1D / max(hann_win_1D(:));

% Apply windows to steering weights
steer_taylor = steer .* taylor_win_2D;
steer_cheb = steer .* cheb_win_2D;
steer_hamming = steer .* hamming_win_2D;
steer_hann = steer .* hann_win_2D;

% Compute Array Factors for windowed beamformers
AF_taylor = zeros(size(THETA_rad));
AF_cheb = zeros(size(THETA_rad));
AF_hamming = zeros(size(THETA_rad));
AF_hann = zeros(size(THETA_rad));

for n = 0:N-1
    for m = 0:M-1
        AF_taylor = AF_taylor + steer_taylor(n+1, m+1) * ...
            exp(1j * 2 * pi * d / lambda * ...
            (n * sin(THETA_rad) .* cos(PHI_rad) + ...
             m * sin(THETA_rad) .* sin(PHI_rad)));
        
        AF_cheb = AF_cheb + steer_cheb(n+1, m+1) * ...
            exp(1j * 2 * pi * d / lambda * ...
            (n * sin(THETA_rad) .* cos(PHI_rad) + ...
             m * sin(THETA_rad) .* sin(PHI_rad)));
        
        AF_hamming = AF_hamming + steer_hamming(n+1, m+1) * ...
            exp(1j * 2 * pi * d / lambda * ...
            (n * sin(THETA_rad) .* cos(PHI_rad) + ...
             m * sin(THETA_rad) .* sin(PHI_rad)));
        
        AF_hann = AF_hann + steer_hann(n+1, m+1) * ...
            exp(1j * 2 * pi * d / lambda * ...
            (n * sin(THETA_rad) .* cos(PHI_rad) + ...
             m * sin(THETA_rad) .* sin(PHI_rad)));
    end
end

AF_taylor_mag = abs(AF_taylor);
AF_taylor_db = 20 * log10(AF_taylor_mag / max(AF_taylor_mag(:)));

AF_cheb_mag = abs(AF_cheb);
AF_cheb_db = 20 * log10(AF_cheb_mag / max(AF_cheb_mag(:)));

AF_hamming_mag = abs(AF_hamming);
AF_hamming_db = 20 * log10(AF_hamming_mag / max(AF_hamming_mag(:)));

AF_hann_mag = abs(AF_hann);
AF_hann_db = 20 * log10(AF_hann_mag / max(AF_hann_mag(:)));

% Calculate Peak-to-Sidelobe Ratios (PSR)
[psr_conv, avg_sidelobe_conv] = calculate_psr(AF_conv_db, theta_desired, phi_desired, theta, phi);
[psr_mvdr, avg_sidelobe_mvdr] = calculate_psr(AF_mvdr_db, theta_desired, phi_desired, theta, phi);
[psr_taylor, avg_sidelobe_taylor] = calculate_psr(AF_taylor_db, theta_desired, phi_desired, theta, phi);
[psr_cheb, avg_sidelobe_cheb] = calculate_psr(AF_cheb_db, theta_desired, phi_desired, theta, phi);
[psr_hamming, avg_sidelobe_hamming] = calculate_psr(AF_hamming_db, theta_desired, phi_desired, theta, phi);
[psr_hann, avg_sidelobe_hann] = calculate_psr(AF_hann_db, theta_desired, phi_desired, theta, phi);

% Display element weights
fprintf('\n=== Conventional Beamformer Weights ===\n');
display_weights(steer, N, M);

fprintf('\n=== MVDR Beamformer Weights ===\n');
display_weights(w_mvdr, N, M);

fprintf('\n=== Taylor Window Beamformer Weights ===\n');
display_weights(steer_taylor, N, M);

fprintf('\n=== Chebyshev Window Beamformer Weights ===\n');
display_weights(steer_cheb, N, M);

fprintf('\n=== Hamming Window Beamformer Weights ===\n');
display_weights(steer_hamming, N, M);

fprintf('\n=== Hann Window Beamformer Weights ===\n');
display_weights(steer_hann, N, M);

% Display performance metrics
fprintf('\n=== Performance Metrics ===\n');
fprintf('Conventional Beamformer:\n');
fprintf('  Peak-to-Sidelobe Ratio: %.2f dB\n', psr_conv);
fprintf('  Average Sidelobe Level: %.2f dB\n', avg_sidelobe_conv);

fprintf('\nMVDR Beamformer:\n');
fprintf('  Peak-to-Sidelobe Ratio: %.2f dB\n', psr_mvdr);
fprintf('  Average Sidelobe Level: %.2f dB\n', avg_sidelobe_mvdr);

fprintf('\nTaylor Window Beamformer:\n');
fprintf('  Peak-to-Sidelobe Ratio: %.2f dB\n', psr_taylor);
fprintf('  Average Sidelobe Level: %.2f dB\n', avg_sidelobe_taylor);

fprintf('\nChebyshev Window Beamformer:\n');
fprintf('  Peak-to-Sidelobe Ratio: %.2f dB\n', psr_cheb);
fprintf('  Average Sidelobe Level: %.2f dB\n', avg_sidelobe_cheb);

fprintf('\nHamming Window Beamformer:\n');
fprintf('  Peak-to-Sidelobe Ratio: %.2f dB\n', psr_hamming);
fprintf('  Average Sidelobe Level: %.2f dB\n', avg_sidelobe_hamming);

fprintf('\nHann Window Beamformer:\n');
fprintf('  Peak-to-Sidelobe Ratio: %.2f dB\n', psr_hann);
fprintf('  Average Sidelobe Level: %.2f dB\n', avg_sidelobe_hann);

% Plot all beampatterns
figure('Position', [100, 100, 1500, 1200]);

% Conventional Beamformer
subplot(2, 3, 1);
imagesc(theta, phi, AF_conv_db);
set(gca, 'YDir', 'normal');
colorbar;
caxis([-40 0]);
title(sprintf('Conventional: PSR=%.1f dB', psr_conv));
xlabel('Elevation θ (deg)');
ylabel('Azimuth φ (deg)');

% MVDR Beamformer
subplot(2, 3, 2);
imagesc(theta, phi, AF_mvdr_db);
set(gca, 'YDir', 'normal');
colorbar;
caxis([-40 0]);
title(sprintf('MVDR: PSR=%.1f dB', psr_mvdr));
xlabel('Elevation θ (deg)');
ylabel('Azimuth φ (deg)');

% Taylor Window Beamformer
subplot(2, 3, 3);
imagesc(theta, phi, AF_taylor_db);
set(gca, 'YDir', 'normal');
colorbar;
caxis([-40 0]);
title(sprintf('Taylor: PSR=%.1f dB', psr_taylor));
xlabel('Elevation θ (deg)');
ylabel('Azimuth φ (deg)');

% Chebyshev Window Beamformer
subplot(2, 3, 4);
imagesc(theta, phi, AF_cheb_db);
set(gca, 'YDir', 'normal');
colorbar;
caxis([-40 0]);
title(sprintf('Chebyshev: PSR=%.1f dB', psr_cheb));
xlabel('Elevation θ (deg)');
ylabel('Azimuth φ (deg)');

% Hamming Window Beamformer
subplot(2, 3, 5);
imagesc(theta, phi, AF_hamming_db);
set(gca, 'YDir', 'normal');
colorbar;
caxis([-40 0]);
title(sprintf('Hamming: PSR=%.1f dB', psr_hamming));
xlabel('Elevation θ (deg)');
ylabel('Azimuth φ (deg)');

% Hann Window Beamformer
subplot(2, 3, 6);
imagesc(theta, phi, AF_hann_db);
set(gca, 'YDir', 'normal');
colorbar;
caxis([-40 0]);
title(sprintf('Hann: PSR=%.1f dB', psr_hann));
xlabel('Elevation θ (deg)');
ylabel('Azimuth φ (deg)');

% --- Helper Functions ---
function [psr, avg_sidelobe] = calculate_psr(af_db, theta_des, phi_des, theta, phi)
    % Find main lobe peak
    [max_val, max_idx] = max(af_db(:));
    [row_idx, col_idx] = ind2sub(size(af_db), max_idx);
    
    % Create mask excluding main lobe (±10 degrees)
    main_lobe_mask = (abs(theta - theta_des) < 10) & (abs(phi - phi_des) < 10);
    sidelobe_mask = ~main_lobe_mask;
    
    % Calculate average sidelobe level
    sidelobes = af_db(sidelobe_mask);
    avg_sidelobe = mean(sidelobes(sidelobes > -40)); % Exclude values at cutoff
    
    % Peak-to-Sidelobe Ratio
    psr = max_val - avg_sidelobe;
end

function display_weights(weights, N, M)
    fprintf('Element\tMagnitude\tPhase(deg)\n');
    fprintf('----------------------------\n');
    for n = 1:N
        for m = 1:M
            mag = abs(weights(n,m));
            phase = angle(weights(n,m)) * 180/pi;
            fprintf('(%d,%d)\t%.4f\t\t%.2f\n', n, m, mag, phase);
        end
    end
end
