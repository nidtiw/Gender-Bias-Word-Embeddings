function [scaledHE, scaledSHE] = gloveMethod(toBeScaled, cooccurrence_matrix, cooccur_words)

% Getting the index of the word to be scaled.
% We will denote 'i' as the HE word and 'j' as the SHE word.
% We will denote 'k' as the gender neutral word to be scaled.
% Worked on by Nidhi Tiwari

isWord = cellfun(@(x)isequal(x,toBeScaled),cooccur_words);
[row_toBeScaled, col_toBeScaled] = find(isWord);

isHe = cellfun(@(x)isequal(x,'he'),cooccur_words);
[row_He, col_He] = find(isHe);

isShe = cellfun(@(x)isequal(x,'she'),cooccur_words);
[row_She, col_She] = find(isShe);

cooccur_mat_rowHe = cooccurrence_matrix(row_He,:);
X_i = sum(cooccur_mat_rowHe);
cooccur_mat_rowShe = cooccurrence_matrix(row_She,:);
X_j = sum(cooccur_mat_rowShe);

X_ik = cooccurrence_matrix(row_He,row_toBeScaled)
X_jk = cooccurrence_matrix(row_She,row_toBeScaled)
X_ij = cooccurrence_matrix(row_He, row_She)

shift = ((X_i*X_jk) - (X_j*X_ik))/(X_i + X_j);
Beta_jk = (X_jk - shift)/X_jk;
Beta_ik = (X_ik + shift)/X_ik;

scaledHE = X_ik * Beta_ik;
scaledSHE = X_jk * Beta_jk;



