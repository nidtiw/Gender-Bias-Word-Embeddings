%{
    EC503 - Learning from Data
    March 2018
    Word Embeddings De-biasing
    Function for soft bias method
    Worked on by: Frank Tranghese
    Requires CVX to be installed
%}

function [Wt] = softDebias(W,g,N)
    % *** INPUTS ***
    % W - entire word embedding set N x D
    % g - gender direction
    % N - gender neutral words we want to debias
    % *** OUTPUTS ***
    % Wt - normalized, linearly transformed word embeddings set
    
    % ensure sparsity
    
    N = sparse(N)';
    
    %get top 50 singular vectors/values
    [U,S,V] = svds(W,50);
    I = sparse(eye(300));
    S = sparse(S);
    V=sparse(V);
    
    %minimize T using CVX solver, available here http://cvxr.com/cvx/
    cvx_begin sdp
    variable X(300,300) symmetric;
    minimize( square_pos(norm(S*V'*(X-I)*V*S,'fro'))+0.2*square_pos(norm(g'*X*N,'fro')));
    subject to
        X == semidefinite(n);
    cvx_end
    
    %decompose X into T*T'
    T = cholcov(X);
    %return normalized transformed W
    Wt = normr(W*T');
    
end