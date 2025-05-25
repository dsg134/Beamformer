% Simulates the beampatterns of an ideal 4x4 phased array with infinite allowable phases in the weights
% Then simulates the beampatterns if the phases are discretely constrained due to digital phase shifting

% -----------------------------------------------
% Input Parameters
f = 2.45e9;              % Frequency in Hz
theta_desired = -30;  % Elevation angle in degrees
phi_desired = 30;     % Azimuth angle in degrees
element_spacing = 0.6; % Element spacing in wavelengths (default 0.5 for λ/2)
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

% Window Function Implementation - Taylor Window
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

% Create 2D windows using outer product
taylor_win_1D = taylorwin(N, 5, -30) * taylorwin(M, 5, -30)';

% Normalize window
taylor_win_2D = taylor_win_1D / max(taylor_win_1D(:));

% Apply window to steering weights
steer_taylor = steer .* taylor_win_2D;

% Compute Array Factor for windowed beamformer
AF_taylor = zeros(size(THETA_rad));
for n = 0:N-1
    for m = 0:M-1
        AF_taylor = AF_taylor + steer_taylor(n+1, m+1) * ...
            exp(1j * 2 * pi * d / lambda * ...
            (n * sin(THETA_rad) .* cos(PHI_rad) + ...
             m * sin(THETA_rad) .* sin(PHI_rad)));
    end
end

AF_taylor_mag = abs(AF_taylor);
AF_taylor_db = 20 * log10(AF_taylor_mag / max(AF_taylor_mag(:)));

% Calculate Peak-to-Sidelobe Ratios (PSR)
[psr_conv, avg_sidelobe_conv] = calculate_psr(AF_conv_db, theta_desired, phi_desired, theta, phi);
[psr_mvdr, avg_sidelobe_mvdr] = calculate_psr(AF_mvdr_db, theta_desired, phi_desired, theta, phi);
[psr_taylor, avg_sidelobe_taylor] = calculate_psr(AF_taylor_db, theta_desired, phi_desired, theta, phi);

% Display element weights with phase wrapping (0 to 360 degrees)
fprintf('\n=== Conventional Beamformer Weights ===\n');
display_weights(steer, N, M);

fprintf('\n=== MVDR Beamformer Weights ===\n');
display_weights(w_mvdr, N, M);

fprintf('\n=== Taylor Window Beamformer Weights ===\n');
display_weights(steer_taylor, N, M);

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

% Plot top three beampatterns
figure('Position', [100, 100, 1500, 400]);

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

% --- Phase Quantization Functions ---
function quantized_phase = quantize_phase(phase_deg)
    available_phases = [1.4, 2.8, 5.6, 11.2, 22.5, 45, 90, 180, 360];
    [~, idx] = min(abs(available_phases - mod(phase_deg, 360)));
    quantized_phase = available_phases(idx);
end

function quantized_weights = quantize_weights(weights)
    quantized_weights = zeros(size(weights));
    for n = 1:size(weights, 1)
        for m = 1:size(weights, 2)
            mag = abs(weights(n,m));
            phase_deg = angle(weights(n,m)) * 180/pi;
            quantized_phase_deg = quantize_phase(phase_deg);
            quantized_weights(n,m) = mag * exp(1j * deg2rad(quantized_phase_deg));
        end
    end
end

% Create quantized versions of the weights
steer_quant = quantize_weights(steer);
w_mvdr_quant = quantize_weights(w_mvdr);
steer_taylor_quant = quantize_weights(steer_taylor);

% Compute Array Factors for quantized beamformers
AF_conv_quant = zeros(size(THETA_rad));
AF_mvdr_quant = zeros(size(THETA_rad));
AF_taylor_quant = zeros(size(THETA_rad));

for n = 0:N-1
    for m = 0:M-1
        % Conventional quantized
        AF_conv_quant = AF_conv_quant + steer_quant(n+1, m+1) * ...
            exp(1j * 2 * pi * d / lambda * ...
            (n * sin(THETA_rad) .* cos(PHI_rad) + ...
             m * sin(THETA_rad) .* sin(PHI_rad)));
        
        % MVDR quantized
        AF_mvdr_quant = AF_mvdr_quant + w_mvdr_quant(n+1, m+1) * ...
            exp(1j * 2 * pi * d / lambda * ...
            (n * sin(THETA_rad) .* cos(PHI_rad) + ...
             m * sin(THETA_rad) .* sin(PHI_rad)));
        
        % Taylor quantized
        AF_taylor_quant = AF_taylor_quant + steer_taylor_quant(n+1, m+1) * ...
            exp(1j * 2 * pi * d / lambda * ...
            (n * sin(THETA_rad) .* cos(PHI_rad) + ...
             m * sin(THETA_rad) .* sin(PHI_rad)));
    end
end

% Convert to dB
AF_conv_quant_mag = abs(AF_conv_quant);
AF_conv_quant_db = 20 * log10(AF_conv_quant_mag / max(AF_conv_quant_mag(:)));

AF_mvdr_quant_mag = abs(AF_mvdr_quant);
AF_mvdr_quant_db = 20 * log10(AF_mvdr_quant_mag / max(AF_mvdr_quant_mag(:)));

AF_taylor_quant_mag = abs(AF_taylor_quant);
AF_taylor_quant_db = 20 * log10(AF_taylor_quant_mag / max(AF_taylor_quant_mag(:)));

% Calculate PSR for quantized patterns
[psr_conv_quant, avg_sidelobe_conv_quant] = calculate_psr(AF_conv_quant_db, theta_desired, phi_desired, theta, phi);
[psr_mvdr_quant, avg_sidelobe_mvdr_quant] = calculate_psr(AF_mvdr_quant_db, theta_desired, phi_desired, theta, phi);
[psr_taylor_quant, avg_sidelobe_taylor_quant] = calculate_psr(AF_taylor_quant_db, theta_desired, phi_desired, theta, phi);

% Display quantized weights
fprintf('\n=== Quantized Conventional Beamformer Weights ===\n');
display_weights(steer_quant, N, M);

fprintf('\n=== Quantized MVDR Beamformer Weights ===\n');
display_weights(w_mvdr_quant, N, M);

fprintf('\n=== Quantized Taylor Window Beamformer Weights ===\n');
display_weights(steer_taylor_quant, N, M);

% Display performance metrics for quantized patterns
fprintf('\n=== Quantized Performance Metrics ===\n');
fprintf('Conventional Beamformer:\n');
fprintf('  Peak-to-Sidelobe Ratio: %.2f dB\n', psr_conv_quant);
fprintf('  Average Sidelobe Level: %.2f dB\n', avg_sidelobe_conv_quant);

fprintf('\nMVDR Beamformer:\n');
fprintf('  Peak-to-Sidelobe Ratio: %.2f dB\n', psr_mvdr_quant);
fprintf('  Average Sidelobe Level: %.2f dB\n', avg_sidelobe_mvdr_quant);

fprintf('\nTaylor Window Beamformer:\n');
fprintf('  Peak-to-Sidelobe Ratio: %.2f dB\n', psr_taylor_quant);
fprintf('  Average Sidelobe Level: %.2f dB\n', avg_sidelobe_taylor_quant);

% Quantized Conventional Beamformer
subplot(2, 3, 4);
imagesc(theta, phi, AF_conv_quant_db);
set(gca, 'YDir', 'normal');
colorbar;
caxis([-40 0]);
title(sprintf('Quantized Conventional: PSR=%.1f dB', psr_conv_quant));
xlabel('Elevation θ (deg)');
ylabel('Azimuth φ (deg)');

% Quantized MVDR Beamformer
subplot(2, 3, 5);
imagesc(theta, phi, AF_mvdr_quant_db);
set(gca, 'YDir', 'normal');
colorbar;
caxis([-40 0]);
title(sprintf('Quantized MVDR: PSR=%.1f dB', psr_mvdr_quant));
xlabel('Elevation θ (deg)');
ylabel('Azimuth φ (deg)');

% Quantized Taylor Window Beamformer
subplot(2, 3, 6);
imagesc(theta, phi, AF_taylor_quant_db);
set(gca, 'YDir', 'normal');
colorbar;
caxis([-40 0]);
title(sprintf('Quantized Taylor: PSR=%.1f dB', psr_taylor_quant));
xlabel('Elevation θ (deg)');
ylabel('Azimuth φ (deg)');

% --- Helper Functions ---
function [psr, max_sidelobe] = calculate_psr(af_db, theta_des, phi_des, theta, phi)
    % Find main lobe peak
    [max_val, max_idx] = max(af_db(:));
    [row_idx, col_idx] = ind2sub(size(af_db), max_idx);
    
    % Create mask excluding main lobe (±10 degrees)
    main_lobe_mask = (abs(theta - theta_des) < 10) & (abs(phi - phi_des) < 10);
    sidelobe_mask = ~main_lobe_mask;
    
    % Find maximum sidelobe level (excluding values at cutoff)
    sidelobes = af_db(sidelobe_mask);
    sidelobes = sidelobes(sidelobes > -40); % Exclude values at cutoff
    max_sidelobe = max(sidelobes);
    
    % Peak-to-Sidelobe Ratio (main lobe peak - largest sidelobe)
    psr = max_val - max_sidelobe;
end

function display_weights(weights, N, M)
    fprintf('Element\tMagnitude\tPhase(deg)\n');
    fprintf('----------------------------\n');
    for n = 1:N
        for m = 1:M
            mag = abs(weights(n,m));
            phase_rad = angle(weights(n,m));
            % Convert to degrees and wrap to [0, 360]
            phase_deg = mod(phase_rad * 180/pi, 360);
            fprintf('(%d,%d)\t%.4f\t\t%.2f\n', n, m, mag, phase_deg);
        end
    end
end
