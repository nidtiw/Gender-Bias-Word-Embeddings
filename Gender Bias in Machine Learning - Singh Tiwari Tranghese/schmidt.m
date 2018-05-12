%{
    EC503 - Learning from Data
    March 2018
    Word Embeddings De-biasing
    Function for Schmidt Method of removing gender subspace
    Worked on by: Frank Tranghese
%}

function [W_schmidt] = schmidt(W,g)
    % *** INPUTS ***
    % W - entire co-occurance matrix of the dataset
    % g - the gender direction
    % *** OUTPUTS ***
    % W_schmidt - entire co-occurance matrix of the dataset with gender
    % subspace removed and normalized
    
    % get vector of the gender components of each word
    Wg = (W*g').*g;
    % subtrack to get genderless wordset
    Wgenderless = W-Wg;
    
    %renormalize and return matrix
    W_schmidt = normr(Wgenderless);
end