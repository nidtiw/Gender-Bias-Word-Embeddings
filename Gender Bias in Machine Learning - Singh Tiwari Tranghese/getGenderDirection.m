%{
    EC503 - Learning from Data
    March 2018
    Word Embeddings De-biasing
    Function for Finding Gender Direction
    Worked on by: Nidhi Tiwari (nidhit@bu.edu)
%}

function [g] = getGenderDirection(wordvecs_norm,word2Index)
% INPUTS
% W - co-occurance matrix of all Word Emebeddings in set (N x d)
% X - co-occurance matrix of all defined by gendered terms
% OUTPUTS
% g - the gender subspace/direction matrix 

%CODE GOES HERE

% Ten word pairs to define the gender direction
directionWords = ["she" "he";"her" "his";"woman" "man";"girl" "boy";"mother" "father";"daughter" "son";"gal" "guy";"female" "male";"her" "his";"Mary" "John"]
% gender_matrix = zeros(10,size(wordvecs_norm,2));
for j = 1:10
    % Getting the index for the words in the female-male pair
    female_idx = word2Index(char(directionWords(j,1)));
    male_idx = word2Index(char(directionWords(j,2)));
    female_vec = wordvecs_norm(female_idx,:);
    male_vec = wordvecs_norm(male_idx,:);
    gender_matrix(j,:) = female_vec - male_vec;
end

[coeff,score,latent,tsquared,explained] = pca(gender_matrix');
covariance = cov(gender_matrix');
[V,D] = eig(covariance);

select_eig = V(:,10);
g = score*select_eig;


figure()
pareto(explained)
xlabel('Principal Component')
ylabel('VarianceExplained (%)')

