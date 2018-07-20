% main function
clear all
format long

% =========== initialize =============
% load data
fid = fopen('bmd_data/bmd_nohead.txt');
celldata = textscan(fid, '%d %f %s %f \n');
gender = string(celldata{3});
N = length(gender);
% seperate male and female
N1 = sum(gender == 'male');
N2 = sum(gender == 'female');
Xo1 = zeros(N1,1); Xo2 = zeros(N2,1);
yo1 = zeros(N1,1); yo2 = zeros(N2,1);
i1=0;  i2=0;
for i=1:1:N
    if(gender(i) == 'male')
      i1 =i1 + 1;
      Xo1(i1) = celldata{2}(i);
      yo1(i1) = celldata{4}(i);
    else
      i2 =i2 + 1;
      Xo2(i2) = celldata{2}(i);
      yo2(i2) = celldata{4}(i);
    end
end
% sort data
Xy1 = [Xo1 yo1];
SXy1 = sortrows(Xy1);
X1 = SXy1(:,1);  % sorted X1
y1 = SXy1(:,2);  % sorted y1
Xy2 = [Xo2 yo2];
SXy2 = sortrows(Xy2);
X2 = SXy2(:,1);  %sorted X2
y2 = SXy2(:,2);  % sorted y2

% Change input to female data
%clearvars N1 X1 y1 Xy1;
%N1 = N2; X1 = X2; y1 = y2; Xy1 = Xy2;

% Build test input matrix X0
N01 = 4*N1;
xmin1 = min(X1);  xmax1 = max(X1);
dx1 = (xmax1 - xmin1) / (N01-1);
X01 = (xmin1 : dx1 : xmax1)';

% ============  (a) Smoothing spline ==============
% Set up N knots at data points X
K1 = N1;
xi1 = X1;  % knots in increasing order

% Training input matrix in natural cubic splines space
B1 = Nat_Cub_Spl(X1, xi1);
% Test input matrix in natural cubic splines space
B01 = Nat_Cub_Spl(X01, xi1);

% compute smoothing splines uisng Eqw. (5.12), (5.13)
lambda = 0.25; 
zs = 1.645;  % Corresponds to 90% confidence
factor = 1e-4;
Omega = Penalty(K1, X1, xi1);
J = inv(B1'*B1 + lambda * Omega + factor*eye(K1) ) * B1'; % Add a small term a*I to remove sigularity
theta1 = J * y1;
y1_pred = B1 *theta1; 
sigma1_sq = sum((y1 - y1_pred).^2) / N1
y01 = B01 * theta1; 
S01 = B01 * J;
Var_y01 = S01 * (sigma1_sq * eye(N1)) * S01';  
errb01 = zs * sqrt(diag(Var_y01));  

% Plot
Plot_spl(X1, y1, X01, y01, errb01, 'blue');

% ============  (b) Bayesian method  ==============
% Use sigma^2 from (a), set Sigma = I
tau = 0.1;
% Predict on test X0 
Mat1 = inv(B1'*B1 + sigma1_sq / tau * eye(K1));  
y01_bayes = B01 * Mat1 * B1' * y1;  
Var_y01_bayes = B01 * Mat1 * B01' * sigma1_sq; 
errb01_bayes = zs * sqrt(diag(Var_y01_bayes)); 

hold on;
Plot_spl(X1, y1, X01, y01_bayes, errb01_bayes, 'cyan');


% ============  (c) Boostrapping  ==============
% Use sigma^2 from (a)
NB = 100; %5; %100;
y1_bstrp = zeros(N1,1);  
y01_avg = zeros(N01,1);
errb01_avg = zeros(N01,1);
sigma1_sq_avg = 0;
color = ['r', 'b', 'k', 'g', 'c'];
i = 0;
figure
% Do smoothing splines in boostrapping samples, and compute the average
for j=1:1:NB
  y1_bstrp = normrnd( y1, sqrt(sigma1_sq) ); % normal distribution around y
  theta1_bstrp = J * y1_bstrp;
  y1_bstrp_pred = B1 *theta1_bstrp; 
  sigma1_sq_bstrp = sum((y1_bstrp - y1_bstrp_pred).^2) / N1;  % Compute error
  y01_bstrp = B01 * theta1_bstrp; 
  y01_avg = y01_avg + y01_bstrp;  % bagging predicted y

  % Plot
  if mod(j,20) == 0
     i = i + 1;
     Plot_multi_curves(X01, y01_bstrp, color(i));
  end

  % Compute varicance of response y
  sigma1_sq_bstrp = sum((y1_bstrp - y1_bstrp_pred).^2) / N1;
  sigma1_sq_avg = sigma1_sq_avg + sigma1_sq_bstrp;  % bagging training error
  % Compute error band for test data, since X is the same for all samples, formular is the same.
  Var_y01_bstrp = S01 * (sigma1_sq_bstrp * eye(N1)) * S01';  % N0-by-N1 * N1-by-N1 * N1-by-N0 = N0-by-N0
  % average error
  errb01_avg = errb01_avg + zs * sqrt(diag(Var_y01_bstrp));  % bagging se
end

y01_avg = y01_avg / NB;
errb01_avg = errb01_avg / NB;
sigma1_sq_avg = sigma1_sq_avg / NB
% Plot
Plot_spl(X1, y1, X01, y01_avg, errb01_avg, 'red');

% ============  (d) 5-flod Cross valication  ==============
Nfold = 5;
N1_cv = N1 - mod(N1, Nfold); % make total number multiple of 5
N1_per_fold = N1_cv / Nfold;
N1f = (Nfold - 1) * N1_per_fold;
Xy1f = zeros(N1f,2); SXy1f = zeros(N1f,2);
X1f = zeros(N1f,1); y1f= zeros(N1f,1);
sigma1f_sq_avg = 0;
y01f_avg = zeros(N01,1);  
errb01f_avg = zeros(N01,1);

figure
for k=1:1:Nfold
   istart = (k-1) * N1_per_fold + 1;  % index for the 4/5 subset
   iend = k * N1_per_fold;
   j = 0;
   for i=1:1:N1_cv
      if i < istart || i > iend
         j = j + 1;
         Xy1f(j,1) = Xy1(i,1);  % pick data points at original (non-sorted) order
         Xy1f(j,2) = Xy1(i,2);
      end
   end 
   SXy1f = sortrows(Xy1f);
   X1f = SXy1f(:,1);  % sorted data in the 4/5 subset
   y1f = SXy1f(:,2);

   % Build smoothing splines basis
   K1f = N1f;
   xi1f = X1f;  % knots in increasing order
   B1f = Nat_Cub_Spl(X1f, xi1f);  % smoothing splines matrix for training
   B1cv = Nat_Cub_Spl(X1, xi1f);  % smoothing splines matrix for cross valication
   B01f = Nat_Cub_Spl(X01, xi1f);  % smoothing splines matrix for test

   % training on the 4/5 subset
   Omegaf = Penalty(K1f, X1f, xi1f);
   Jf = inv(B1f' * B1f + lambda * Omegaf + factor*eye(K1f) ) * B1f'; % K1f-by-N1f
   theta1f = Jf * y1f;  % K1f-by-N1f * N1f-by-1 = K1f-by-1

   % compute cross valication error
   y1f_pred = B1cv * theta1f;  % Preidict on origina x
   sigma1f_sq = sum((y1 - y1f_pred).^2) / N1; % Cross valication error: sum of all oriignal points
   sigma1f_sq_avg = sigma1f_sq_avg + sigma1f_sq;  % bagging CV error

   % Predict at test points
   y01f = B01f * theta1f;  
   y01f_avg = y01f_avg + y01f;  % bagging predicted y for all 4/5 subsets
   Plot_multi_curves(X01, y01f, color(k));
  
   % compute error bands
   S01f = B01f * Jf;   % S_lambda
   Var_y01f = S01f * (sigma1f_sq * eye(N1f)) * S01f';  % Var(y_hat)
   errb01f = zs * sqrt(diag(Var_y01f));  % standard deviation
   errb01f_avg = errb01f_avg + errb01f;  % bagging se
  
end

y01f_avg = y01f_avg / Nfold;
errb01f_avg = errb01f_avg / Nfold;
sigma1f_sq_avg = sigma1f_sq_avg / Nfold
%Plot
Plot_spl(X1, y1, X01, y01f_avg, errb01f_avg, 'black');

% Plot together
figure
plot(X01,y01,'blue', X01,y01_bayes,'cyan', X01,y01_avg,'red', X01,y01f_avg,'black');
xlabel('age')
ylabel('BMD')


