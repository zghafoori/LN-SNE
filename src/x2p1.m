function [beta, D] = x2p1(X, u, tol, verbos)
%X2P Identifies appropriate sigma's to get kk NNs up to some tolerance
%
%   [P, beta] = x2p(xx, kk, tol)
%
% Identifies the required precision (= 1 / variance^2) to obtain a Gaussian
% kernel with a certain uncertainty for every datapoint. The desired
% uncertainty can be specified through the perplexity u (default = 15). The
% desired perplexity is obtained up to some tolerance that can be specified
% by tol (default = 1e-4).
% The function returns the final Gaussian kernel in P, as well as the
% employed precisions per instance in beta.
%


if ~exist('u', 'var') || isempty(u)
    u = 15;
end
if ~exist('tol', 'var') || isempty(tol)
    tol = 1e-4;
end

if nargin < 4, verbos = 1; end;

% Initialize some variables
n = size(X, 1);                     % number of instances
beta = ones(n, 1);                  % empty precision vector
logU = log(u);                      % log of perplexity (= entropy)

% Compute pairwise distances
if verbos, disp('Computing pairwise distances...'); end;
sum_X = sum(X .^ 2, 2);
D = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * X * X'));
D(1:size(D,1)+1:end) = 0;
[~,ixx] = max(mean(D,2));
Di = D(ixx, [1:ixx-1 ixx+1:end]);
[beta(ixx)] = binSearch(beta(ixx),Di,logU,tol,50);

% [~,inds] = sort(beta);
% for i = 1:floor(0.05*size(X,1))
%     Di = D(inds(i), [1:inds(i)-1 inds(i)+1:end]);
%     [beta(inds(i))] = binSearch(beta(inds(i)),Di,logU,tol,200);
% end;
beta = beta(ixx);
end


function betai = binSearch(betai,Di,logU,tol,triesNo)
H = Hbeta(Di, betai);
Hdiff = H - logU;
tries = 0;
% Set minimum and maximum values for precision
betamin = -Inf;
betamax = Inf;
while abs(Hdiff) > tol && tries < triesNo    
    % If not, increase or decrease precision
    if Hdiff > 0
        betamin = betai;
        if isinf(betamax)
            betai = betai * 2;
        else
            betai = (betai + betamax) / 2;
        end
    else
        betamax = betai;
        if isinf(betamin)
            betai = betai / 2;
        else
            betai = (betai + betamin) / 2;
        end
    end
    
    % Recompute the values
    H = Hbeta(Di, betai);
    Hdiff = H - logU;
    tries = tries + 1;
end
end



% Function that computes the Gaussian kernel values given a vector of
% squared Euclidean distances, and the precision of the Gaussian kernel.
% The function also computes the perplexity of the distribution.
function H = Hbeta(D, beta)
P = exp(-D * beta);
sumP = sum(P);
H = log(sumP) + beta * sum(D .* P) / sumP;
end


