%% Initialisation

clear;
close all;

% Loading toolbox
init_unlocbox();
ltfatstart();

verbose = 2;    % verbosity level

% Regularization parameter: weight of the fielity term
tau = 50;
% Noise level
sigma = 0.1;
% Percent of missing pixels
p = 50;

%% Defining the problem


% Original image
im_original = checkerboard();

% Depleted image
mask = rand(size(im_original))>p/100;
z = mask .* im_original + sigma * rand(size(im_original));



%% Defining proximal operators

% Define the wavelet operator
L = @(x)  fwt2(x,'db8',6);
Lt = @(x)  ifwt2(x,'db8',6);

% setting the function tau * || Mx - y ||_2^2  
f1.proxL = @(x, T) (1+tau*T*mask).^(-1) .* (Lt(x)+tau*T*mask.*z);
f1.eval = @(x) tau * norm(mask .* x - z)^2;

% setting the function || L x ||_1 using ADMM to move the operator ot of
% the proximal
param_l1.verbose = verbose - 1;
f2.prox = @(x, T) prox_l1(x, T, param_l1);
f2.eval = @(x) norm(L(x),1);
f2.L = L;
f2.Lt = Lt;
f2.norm_L = 1;

%% solving the problem

% setting different parameter for the solver
paramsolver.verbose = verbose;     % display parameter
paramsolver.maxit = 100;           % maximum number of iterations
paramsolver.tol = 1e-3;            % tolerance to stop iterating
paramsolver.gamma = 1;             % stepsize
% Activate debug mode in order to compute the objective function at each
% iteration.
paramsolver.debug_mode = 1; 
fig=figure(100);
paramsolver.do_sol=@(x) plot_image(x,fig);  

sol = admm(z, f1, f2, paramsolver);

%% displaying the result
imagesc_gray(im_original, 1, 'Original image');
imagesc_gray(z, 2, 'Depleted image');
imagesc_gray(sol, 3, 'Reconstructed image');
    
%% Closing the toolbox
close_unlocbox();

