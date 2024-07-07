function timeseries = Robinson_state_dependent_noise(noises,coupling,h,T,NN,sc_matrix,delay_matrix,Ksi)
% Hindriks/Tewarie 2022 simulate Robinson model with state dependent noise
% Dissociation between phase and power correlation networks in
% the human brain can be explained by co-occurrent bursts

% input parameters:
% noises        - mean noise level
% coupling      - coupling parameter between two populations
% h             - integration time steo
% T             - simulation time in seconds
% NN            - number of populations or nodes
% sc_matrix     - structural connectivity matrix (NN x NN)
% delay_matrix  - include heterogenous delays (NN x NN)
% Ksi           - state dependent noise parameter

% output parameters
% timeseries    - simulated timeseries (firing rate cortical excitatory
% population)

%% model parameters
Q_max = 250;  % 1/sec
theta = 15;   % mV
sigma = 3.3;  % mV
g =  100;     % cortical damping (1/sec)
a_e = 50;     % cortical EPSP decay rate (1/sec)
b_e = 200;    % cortical EPSP rise rate (1/sec)
a_i = 50;     % cortical IPSP decay rate (1/sec)
b_i = 200;    % cortical IPSP rise rate (1/sec)
a_t = 50;     % thalamic PSP decay rate (1/sec)
b_t = 200;    % thalamic PSP rise rate (1/sec)
v_ee =  1.2;  % mVs
v_ei = -1.8;  % mVs
v_es =  1.2;  % mVs
v_ii = -1.8;  % mVs
v_ie =  1.2;  % mVs
v_is =  1.2;  % mVs
v_sr = -0.8;  % mVs
v_se =  1.2;  % mVs
v_rs =  0.2;  % mVs
v_re =  0.4;  % mVs
v_sn =  0.5;    % mVs 
v_ee_ext =  coupling; %coupling; %mVs % 0.84
q =     0; %mV (q := v_sn*phi_n) constant noise level %1.2
q_std = noises;   % 0.13; %noise std 0.1
tau =   0.04; % delay between cortex and thalamus  
delay_matrix = delay_matrix./h;
tau_c = max(max(delay_matrix)); % maximum cortical delay

%simulation parameters
N = T/h + max(ceil(tau/h) + ceil(tau_c)); %number of samples

% initialization of some state variables
X = zeros(10,NN,N); % initialization of state vector
phi_delay = zeros(1,NN);
% phi_delay_sum = zeros(1,NN);
index = 6; % noise index


%% main loop
for n=tau/h + ceil(tau_c) + 1:N % loop over time n
     
     % delays, based on distance between Euclidean distance between
     % regions/velocity
     for f=1:size(sc_matrix,2)
        for k = 1:size(sc_matrix,2)
            if k~=f
                delay = delay_matrix(k,f);
                phi_delay(k,f) = X(9,k,n-ceil(delay)) .* sc_matrix(k,f);
%                 phi_delay(k) = X(9,k,n-ceil(delay)) .* sc_matrix(k,f);

            end
        end
%         phi_delay_sum(f) = mean(phi_delay);
     end
     phi_delay_sum = mean(phi_delay);
     
     % noise in the model
     noise = zeros(10,NN);
     noise(index,:) = v_sn.*((a_t*b_t*q_std*randn(1,NN) + a_t*b_t*q_std*randn(1,NN) .* (Ksi*(X(9,:,n-tau/h))))); % state dependent noise
%      noise(index,:) = a_e*b_e*q_std*randn(1,NN); % normal noise input
     
     % main computation, Euler-Maruyama
     X(:,:,n+1) = X(:,:,n) + (h.*dynamics(X(:,:,n),X(:,:,n-tau/h),phi_delay_sum)) + sqrt(h)*noise;
   
end
timeseries = squeeze(X(9,:,:));
timeseries(:,1:2/h)=[]; % delete first two seconds


%% differential equations Robinson
% set laplace operator to zero (model only time dependent)
function dX = dynamics(X,X_delay,sumPhe)
dX(1,:)  = X(2,:); % V_e^1
dX(2,:)  = a_e*b_e*(-X(1,:) + v_ee*X(9,:) + v_ee_ext*sumPhe + v_ei*S(X(3,:)) + v_es*S(X_delay(5,:))) - (a_e+b_e)*X(2,:);
dX(3,:)  = X(4,:); % V_i^1
dX(4,:)  = a_i*b_i*(-X(3,:) + v_ie*X(9,:) + v_ii*S(X(3,:)) + v_is*S(X_delay(5,:))) - (a_i+ b_i)*X(4,:);
dX(5,:)  = X(6,:); % V_s^1
dX(6,:) = a_t*b_t*(-X(5,:) + q + v_se*X_delay(9,:) + v_sr*S(X(7,:))) - (a_t + b_t)*X(6,:);
dX(7,:) = X(8,:);  % V_r
dX(8,:) = a_t*b_t*(-X(7,:) + v_re*X_delay(9,:) + v_rs*S(X(5,:))) - (a_t + b_t)*X(8,:);
dX(9,:) = X(10,:); % phi_e
dX(10,:) = g^2*(-X(9,:) + S(X(1,:))) - 2*g*X(10,:);
end


%% sigmoid function to convert potential to firing rate
function output = S(v)
output = Q_max./(1 + exp(-(v - theta)./sigma));
end

end
