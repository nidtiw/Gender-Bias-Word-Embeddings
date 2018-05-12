%{
    EC503 - Learning from Data
    March 2018
    Word Embeddings De-biasing
    Function for quantifying direct bias
    Worked on by: Frank Tranghese
%}

function [d] = directBias(W,g,c)
% INPUTS
% W - co-occurance matrix of words to test for direct bias (n x d)
% g - the gender direction, 1xd
% c - strictness parameter (most strict = 0, Bolukbasi uses 1?)
% OUTPUTS
% d - the matrix of quanitifed direct bias for all words in w (n x 1)

% get length of W
N = size(W,1); % assumes words in w represented by each row

%find cosine similarity of all words = <w,g> / ||w||*||g|| where w is each
%words in W
cos_vec = W*g' ./ ((vecnorm(W,2,2))*norm(g,2));

d = (1/N) * sum((abs(cos_vec)).^c);

end