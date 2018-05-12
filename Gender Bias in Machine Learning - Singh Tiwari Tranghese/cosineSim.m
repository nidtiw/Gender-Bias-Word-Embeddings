%{
    EC503 - Learning from Data
    March 2018
    Word Embeddings De-biasing
    Function for quantifying direct bias
    Worked on by: Frank Tranghese
%}

function [sim] = cosineSim(W,V)
% INPUTS
% W - co-occurance matrix of words to test for similarity (n x d)
% V - co-occurance matrix of words we want to test similarity with
% OUTPUTS
% sim - similarity between words in W and V


%find cosine similarity of all words = <w,g> / ||w||*||g|| where w is each
%words in W
sim = W*V' ./ (vecnorm(W,2,2)*vecnorm(V,2,2));


end