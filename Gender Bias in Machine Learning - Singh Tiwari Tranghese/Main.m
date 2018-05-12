%{
    EC503 - Learning from Data
    March 2018
    Word Embeddings De-biasing
    Main script
    Work by: Frank Tranghese, Nidhi Tiwari, Aditya Singh
%}

% CODE GOES HERE
% Get the word vectors for the dataset
% Loading the co-occurrence matrix
load('coo_matrix'); 
% Loading the 6000 wordslist which the co-occurrence matrix comprises of.
load('final_6kwords.mat');
fileID = fopen('w2v_gnews_small.txt');
fmt = ['%s ' repmat('%f ',1,300)];
words = textscan(fileID,fmt);
fclose('all');
%% The idea to map the words with the word vectors using a containerised map data structure...
%% was taken by Chris Mccormick's word2vec implementation which can be found:
%% here: https://github.com/chrisjmccormick/word2vec_matlab 
wordvecs = cell2mat(words(:,2:end));
words = words(:,1);
words_part = {};
for p = 1:length(words{:})
    words_part{p,1} = words{1,1}{p,1};
end
wordIndex = containers.Map(words_part, (1:length(words_part)));
% Normalising the word vectors
norms = zeros(size(wordvecs,1),size(wordvecs,2));
for j = 1:size(wordvecs,1)
    vector = wordvecs(j,:);
    % Normalising
    norms(j,:) = vector./norm(vector);
end
% Getting the equalise word pairs and get the vectors
fname = 'equalize_wordpairs.json'; 
fid = fopen(fname); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 
equalize = jsondecode(str);

% Getting the gender direction
g = getGenderDirection(norms,wordIndex);

% Getting the normalised vectors of the occupation words list
[occupationWords, occupationVectors] = getVectorsOfType('occupations.json',norms,wordIndex,words_part);
% Getting the normalised vectors of the gender specific words list
[genderspecWords, genderspecVectors] = getGenderSpecific('gender_specific.json',norms,wordIndex,words_part);
%% *** Schmidt Method ***

Wschmidt = schmidt(sparse(norms),g');

%% get occupations and gender specific again

[occupationWordsSchmidt,occupationVectorsSchmidt] = getVectorsOfType('occupations.json',Wschmidt,wordIndex,words_part);
[genderspecWordsSchmidt,genderspecVectorsSchmidt] = getGenderSpecific('gender_specific.json',Wschmidt,wordIndex,words_part);

%% ** Get Direct and Indirect Bias **
directOccSchmidt = directBias(occupationVectorsSchmidt,g',1)
directSpecSchmidt = directBias(genderspecVectorsSchmidt,g',1)

%% ** Get Test Words ***

heSchmidt = Wschmidt(wordIndex('he'),:);
sheSchmidt = Wschmidt(wordIndex('she'),:);
footballSchmidt = Wschmidt(wordIndex('football'),:);
softballSchmidt = Wschmidt(wordIndex('softball'),:);
nurseSchmidt = Wschmidt(wordIndex('nurse'),:);
receptionistSchmidt = Wschmidt(wordIndex('receptionist'),:);
pinkSchmidt = Wschmidt(wordIndex('pink'),:);

he = norms(wordIndex('he'),:);
she = norms(wordIndex('she'),:);
football = norms(wordIndex('football'),:);
softball = norms(wordIndex('softball'),:);
nurse = norms(wordIndex('nurse'),:);
receptionist = norms(wordIndex('receptionist'),:);
pink = norms(wordIndex('pink'),:);
scientist = norms(wordIndex('scientist'),:);

%% Indirect Bias Test

MeanIndirectFootballSchmidt = mean(indirectBias(occupationVectorsSchmidt,footballSchmidt,g'))
MeanIndirectSoftballSchmidt = mean(indirectBias(occupationVectorsSchmidt,softballSchmidt,g'))

indirectReceptionistSoftballSchmidt = indirectBias(receptionistSchmidt,softballSchmidt,g')
indirectReceptionistPinkSchmidt = indirectBias(receptionistSchmidt,pinkSchmidt,g')
indirectNurseSoftballSchmidt = indirectBias(nurseSchmidt,softballSchmidt,g')
indirectNursePinkSchmidt = indirectBias(nurseSchmidt,pinkSchmidt,g')
%% *** Graph on He-She Similarity Plane

heSimOcc = cosineSim(occupationVectorsSchmidt,heSchmidt);
sheSimOcc = cosineSim(occupationVectorsSchmidt,sheSchmidt);
heSimSpec = cosineSim(genderspecVectorsSchmidt,heSchmidt);
sheSimSpec = cosineSim(genderspecVectorsSchmidt,sheSchmidt);

figure;
plot(heSimOcc,sheSimOcc,'ob');
refline(1,0);
xlabel('Similiarty to He (Radians)');
ylabel('Similarity to She (Radians)');
legend('Occupations','Neutrality Line','Location','Southeast')

figure;
plot(heSimSpec,sheSimSpec,'ob');
refline(1,0);
xlabel('Similiarty to He (Radians)');
ylabel('Similarity to She (Radians)');
legend('Occupations','Neutrality Line','Location','Southeast')

%% *** top Indirect Bias Occupations for Schmidt****

[topIndirectFootballSchmidt,indTopIndirectFootballSchmidt] = maxk(indirectBias(occupationVectorsSchmidt,footballSchmidt,g'),5);
[MeanIndirectSoftballSchmidt,indTopIndirectSofttballSchmidt] = maxk(indirectBias(occupationVectorsSchmidt,softballSchmidt,g'),5);

%% *** Soft Debias Method ***

%example of how to use, commented out due to long runtime
%Wsoft = softDebias(sparse(norms),g');

%load pre-solved transform
load T.mat

Wsoft = normr(norms*T');

%% get occupations and gender specific again Soft Debias

[occupationWordsSoft,occupationVectorsSoft] = getVectorsOfType('occupations.json',Wsoft,wordIndex,words_part);
[genderspecWordsSoft,genderspecVectorsSoft] = getGenderSpecific('gender_specific.json',Wsoft,wordIndex,words_part);

%% ** Get Direct and Indirect Bias for Soft Debias **
directOccSoft = directBias(occupationVectorsSoft,g',1)
directSpecSoft = directBias(genderspecVectorsSoft,g',1)

%% ** Get Test Words for Soft Debias***

heSoft = Wsoft(wordIndex('he'),:);
sheSoft = Wsoft(wordIndex('she'),:);
footballSoft = Wsoft(wordIndex('football'),:);
softballSoft = Wsoft(wordIndex('softball'),:);
nurseSoft = Wsoft(wordIndex('nurse'),:);
receptionistSoft = Wsoft(wordIndex('receptionist'),:);
pinkSoft = Wsoft(wordIndex('pink'),:);

he = norms(wordIndex('he'),:);
she = norms(wordIndex('she'),:);
football = norms(wordIndex('football'),:);
softball = norms(wordIndex('softball'),:);
nurse = norms(wordIndex('nurse'),:);
receptionist = norms(wordIndex('receptionist'),:);
pink = norms(wordIndex('pink'),:);
scientist = norms(wordIndex('scientist'),:);

%% Indirect Bias Test for Soft Debias

MeanIndirectFootballSoft = mean(indirectBias(occupationVectorsSoft,footballSoft,g'))
MeanIndirectSoftballSoft = mean(indirectBias(occupationVectorsSoft,softballSoft,g'))

indirectReceptionistSoftballSoft = indirectBias(receptionistSoft,softballSoft,g')
indirectReceptionistPinkSoft = indirectBias(receptionistSoft,pinkSoft,g')
indirectNurseSoftballSoft = indirectBias(nurseSoft,softballSoft,g')
indirectNursePinkSoft = indirectBias(nurseSoft,pinkSoft,g')
%% *** Graph on He-She Similarity Plane for Soft Debias

heSimOcc = cosineSim(occupationVectorsSoft,heSoft);
sheSimOcc = cosineSim(occupationVectorsSoft,sheSoft);
heSimSpec = cosineSim(genderspecVectorsSoft,heSoft);
sheSimSpec = cosineSim(genderspecVectorsSoft,sheSoft);

figure;
plot(heSimOcc,sheSimOcc,'ob');
refline(1,0);
xlabel('Similiarty to He (Radians)');
ylabel('Similarity to She (Radians)');
legend('Occupations','Neutrality Line','Location','Southeast')

figure;
plot(heSimSpec,sheSimSpec,'ob');
refline(1,0);
xlabel('Similiarty to He (Radians)');
ylabel('Similarity to She (Radians)');
legend('Occupations','Neutrality Line','Location','Southeast')

%% *** top Indirect Bias Occupations for Soft Debias****

[topIndirectFootballSchmidt,indTopIndirectFootballSchmidt] = maxk(indirectBias(occupationVectorsSchmidt,footballSchmidt,g'),5);
[MeanIndirectSoftballSchmidt,indTopIndirectSofttballSchmidt] = maxk(indirectBias(occupationVectorsSchmidt,softballSchmidt,g'),5);


%% *** Carry out the Hard Debias Method ****
interim2 = {};
interim3 = {};
for k = 1:length(equalize)
    interim2{k,1} = char(equalize{k,1}(1,1));
    interim3{k,1} = char(equalize{k,1}(2,1));
end
y = keys(wordIndex);
arr3 = ismember(interim2,y);
arr4 = ismember(interim3,y);
counter3 = 0;
for i = 1:length(interim2)
    if(arr3(i)==1 && arr4(i)==1)
        counter3 = counter3 + 1;
        equalize_idx(counter3) = i
        equalize_fin(counter3,:,1) = norms(wordIndex(interim2{i,1}),:);
        equalize_fin(counter3,:,2) = norms(wordIndex(interim3{i,1}),:);
    end
end
equalize_total(:,:) = [equalize_fin(:,:,1);equalize_fin(:,:,2)];
% Carrying out Hard Debias on the occupation words and the Equalise words
[re_embeddings_occupations, re_embeddings_equalise] = hardDebias(occupationVectors, equalize_fin,g)

% Plotting the occupation words He-She similarity before debais
occUnbiased_simHe = (occupationVectors*he') ./ (vecnorm(occupationVectors,2,2)*vecnorm(he,2,2));
occUnbiased_simShe = (occupationVectors*she') ./ (vecnorm(occupationVectors,2,2)*vecnorm(she,2,2));
figure;
plot(occUnbiased_simHe,occUnbiased_simShe,'o','Color',[0,0,1],'MarkerSize',3)
hold on
refline(1,0)
legend('Occupations','Neutrality','Location','Southeast')
title('Occupations Similarity (He-She) before Hard-debiasing')
xlabel('Similarity to He(Radians)')
ylabel('Similarity to She(Radians)')

% Plotting the occupation words He-She similarity after debais
occ_reembed_simHe = (re_embeddings_occupations*he') ./ (vecnorm(re_embeddings_occupations,2,2)*vecnorm(he,2,2));
occ_reembed_simShe = (re_embeddings_occupations*she') ./ (vecnorm(re_embeddings_occupations,2,2)*vecnorm(she,2,2));
figure;
plot(occ_reembed_simHe,occ_reembed_simShe,'o','Color',[0,0,1],'MarkerSize',3)
hold on
refline(1,0)
legend('Occupations','Neutrality','Location','Southeast')
title('Occupations Similarity (He-She) after Hard-debiasing')
xlabel('Similarity to He(Radians)')
ylabel('Similarity to She(Radians)')

% Testing the direct bias of the occupation words after re-embedding
occ_DB_after = directBias(re_embeddings_occupations,g',1);

% Testing the direct bias of the gendered equalise words after
% re-embedding
% Before debiasing:
spec_DB_before = directBias(equalize_total,g',1);
% After debiasing:
spec_DB_after = directBias(re_embeddings_equalise,g',1);

% Testing the indirect bias of 'nurse' and 'scientist' with respect to
% football and softball 
isNurse = cellfun(@(x)isequal(x,'nurse'),occupationWords);
[row, col] = find(isNurse);
nurse_reembed_vec = re_embeddings_occupations(row,:);

isScientist = cellfun(@(x)isequal(x,'scientist'),occupationWords);
[row, col] = find(isScientist);
scientist_reembed_vec = re_embeddings_occupations(row,:);

nurse_IDBfootball_after = indirectBias(nurse_reembed_vec,football,g');
nurse_IDBsoftball_after = indirectBias(nurse_reembed_vec,softball,g');

scientist_IDBfootball_after = indirectBias(scientist_reembed_vec,football,g');
scientist_IDBsoftball_after = indirectBias(scientist_reembed_vec,softball,g');

% Carrying out the Glove method. 
matrix_part(matrix_part == 0) = 1*10^-10;
wordsCooccur = cellstr(data);
[scaledW1He, scaledW1She] = gloveMethod('Scientist', matrix_part, wordsCooccur)
[scaledW2He, scaledW2She] = gloveMethod('Laboratory,', matrix_part, wordsCooccur)
reScaled_MatrixPart = matrix_part;

% Evaluating the GloVe method for the co-occurrence matrix 
isHe = cellfun(@(x)isequal(x,'he'),wordsCooccur);
[row_He, col_He] = find(isHe);
isShe = cellfun(@(x)isequal(x,'she'),wordsCooccur);
[row_She, col_She] = find(isShe);
isW1 = cellfun(@(x)isequal(x,'Scientist'),wordsCooccur);
[row_W1, col_W1] = find(isW1);
W1_beforeScaling_He = matrix_part(row_He,row_W1);
W1_beforeScaling_She = matrix_part(row_She,row_W1);

isW2 = cellfun(@(x)isequal(x,'Laboratory,'),wordsCooccur);
[row_W2, col_W2] = find(isW2);
W2_beforeScaling_He = matrix_part(row_He,row_W2);
W2_beforeScaling_She = matrix_part(row_She,row_W2);


reScaled_MatrixPart(row_W1,row_He) = scaledW1He;
reScaled_MatrixPart(row_He,row_W1) = scaledW1He;
reScaled_MatrixPart(row_W1,row_She) = scaledW1She;
reScaled_MatrixPart(row_She,row_W1) = scaledW1She;

reScaled_MatrixPart(row_W2,row_He) = scaledW2He;
reScaled_MatrixPart(row_He,row_W2) = scaledW2He;
reScaled_MatrixPart(row_W2,row_She) = scaledW2She;
reScaled_MatrixPart(row_She,row_W2) = scaledW2She;

% Getting Xi and Xj for 'he'and 'she' after rescaling. 
cooccur_mat_rowHe_before = matrix_part(row_He,:);
X_i_before = sum(cooccur_mat_rowHe_before);
cooccur_mat_rowShe_before = matrix_part(row_She,:);
X_j_before = sum(cooccur_mat_rowShe_before);

P_scientist_he_beforeScaling = W1_beforeScaling_He/X_i_before;
P_scientist_she_beforeScaling = W1_beforeScaling_She/X_j_before;
before_W1 = P_scientist_he_beforeScaling/P_scientist_she_beforeScaling

P_W2_he_beforeScaling = W2_beforeScaling_He/X_i_before;
P_W2_she_beforeScaling = W2_beforeScaling_She/X_j_before;
before_W2 = P_W2_he_beforeScaling/P_W2_she_beforeScaling;

cooccur_mat_rowHe_after = reScaled_MatrixPart(row_He,:);
X_i_after = sum(cooccur_mat_rowHe_after);
cooccur_mat_rowShe_after = reScaled_MatrixPart(row_She,:);
X_j_after = sum(cooccur_mat_rowShe_after);

P_scientist_he_afterScaling = scaledW1He/X_i_after;
P_scientist_she_afterScaling = scaledW1She/X_j_after;
after_W1 = P_scientist_he_afterScaling/P_scientist_she_afterScaling

P_W2_he_afterScaling = scaledW2He/X_i_after;
P_W2_she_afterScaling = scaledW2She/X_j_after;
after_W2 = P_W2_he_afterScaling/P_W2_she_afterScaling;

