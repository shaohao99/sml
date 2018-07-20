function dkx = Compute_dkx(k, K, x, xi)

  if k > K - 1
     disp('Error: k must be not larger than K-1.')
  end

  N = length(x);
  dkx = zeros(N, 1);  
  inv_inteval = 1.0 / (xi(K) - xi(k));
  for i=1:1:N
    if x(i) > xi(K)
      dkx(i) = dkx(i) - (x(i)-xi(K))^3;
    end
    if x(i) > xi(k)
      dkx(i) = (dkx(i) + (x(i)-xi(k))^3) * inv_inteval ;
    end
  end

end
