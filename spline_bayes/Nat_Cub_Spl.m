% return natural cubic splines for given K knots xi(k), k=1, 2, ..., K
function S = Nat_Cub_Spl(x, xi) 

  K = length(xi); %+ 1;  % number of splines = number of knots
  N = length(x);
  S = zeros(N, K);  % spline matrix spanded by x
  
  S(:,1) = ones(N,1);
  S(:,2) = x(:);

  d_Km1 = zeros(N, 1);  
  Km1 = K - 1;
  d_Km1 = Compute_dkx(Km1, K, x, xi); % compute d_K-1

  dkx = zeros(N, 1); 
  for k=1:1:K-2
      dkx = Compute_dkx(k, K, x, xi); % compute d_k
      S(:,k+2) =  dkx(:) - d_Km1(:);  % compute the k+2-th spline
  end

end
