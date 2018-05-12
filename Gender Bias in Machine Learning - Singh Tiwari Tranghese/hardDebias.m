function [re_embeddings_occupations, re_embeddings_equalize] = hardDebias(occupations, equalize_fin, gender_direction)
%{
    EC503 - Learning from Data
    March 2018
    Word Embeddings De-biasing
    Function for Bias Correction (Hard Be-Bias Method)
    Worked on by: Nidhi Tiwari
%}

% Getting the component of the occupation words in the gender direction
occ_g = (occupations*gender_direction).*gender_direction';
% Step 1: Neutralise the occupation words in the gender subspace
re_embeddings_occupations = zeros(size(occupations,1),300);
for d = 1:size(occupations,1)
    re_embeddings_occupations(d,:) = (occupations(d,:) - occ_g(d,:))./norm(occupations(d,:) - occ_g(d,:));
end

% Step 2: Make the equalise wordsets equidistant to the words in the gender
% subspace.
arg1 = (equalize_fin(:,:,1) + equalize_fin(:,:,2))./2;
arg2 = (gender_direction*ones(1,45))';

nu = arg1 - (arg2 .* (dot(arg1,arg2,1)/dot(gender_direction,gender_direction)));
EQ_g1(:,:) = (equalize_fin(:,:,1)*gender_direction).*gender_direction';
EQ_g2(:,:) = (equalize_fin(:,:,2)*gender_direction).*gender_direction';
wb_E1 = (EQ_g1 - (dot(arg1,arg2,1)/dot(gender_direction,gender_direction)))./norm(EQ_g1 - (dot(arg1,arg2,1)/dot(gender_direction,gender_direction)));
wb_E2 = (EQ_g2 - (dot(arg1,arg2,1)/dot(gender_direction,gender_direction)))./norm(EQ_g2 - (dot(arg1,arg2,1)/dot(gender_direction,gender_direction)));
re_embed_E1(:,:) = nu + (sqrt(1-(sum(nu.^2,2))).*wb_E1);
re_embed_E2(:,:) = nu + (sqrt(1-(sum(nu.^2,2))).*wb_E2);

re_embeddings_equalize = [re_embed_E1;re_embed_E2];


