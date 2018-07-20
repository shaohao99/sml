% return natural cubic splines for given K knots xi(k), k=1, 2, ..., K
function SD2 = Nat_Cub_Spl_D2(x, xi) 

  K = length(xi); %+ 1;  % number of splines = number of knots
  N = length(x);
  SD2 = zeros(N, K);  % spline matrix spanded by x
  
  dkx_d2_Km1 = zeros(N, 1);  
  Km1 = K - 1;
  dkx_d2_Km1 = Compute_dkx_d2(Km1, K, x, xi); % compute d_K-1

  dkx_d2 = zeros(N, 1); 
  for k=1:1:K-2
      dkx_d2 = Compute_dkx_d2(k, K, x, xi); % compute d_k
      SD2(:,k+2) =  dkx_d2(:) - dkx_d2_Km1(:);  % compute the k+2-th spline
  end

end
