function dkx_d2 = Compute_dkx_d2(k, K, x, xi)

  if k > K - 1
     disp('Error: k must be not larger than K-1.')
  end

  N = length(x);
  dkx_d2 = zeros(N, 1);  
  inv_interval6 = 6.0 / (xi(K) - xi(k));
  for i=1:1:N
    if x(i) > xi(K)
      dkx_d2(i) = dkx_d2(i) - (x(i)-xi(K));
    end
    if x(i) > xi(k)
      dkx_d2(i) = (dkx_d2(i) + (x(i)-xi(k)) ) * inv_interval6; 
    end
  end
  %dkx_d2

end
