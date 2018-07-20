% Compute penalty term
function Omega = Penalty(K, x, xi)

   N = length(x);
   Omega = zeros(K,K);
    
   % Compute grid stride
   dx = diff(x);  % diff is N-1 size

   % Compute penalty term
   SD2 = Nat_Cub_Spl_D2(x, xi);

   for j=1:1:K
       for k=1:1:K
           Omega(j,k) = Omega(j,k) + sum( SD2(1:N-1,j) .* SD2(1:N-1,k) .* dx(:));
       end
   end

end
