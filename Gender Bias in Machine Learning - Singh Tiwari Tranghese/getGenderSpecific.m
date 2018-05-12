%{
    EC503 - Learning from Data
    March 2018
    Word Embeddings De-biasing
    Function for Finding Vectors of Type passed
    Worked on by: Nidhi Tiwari & Frank Tranghese
%}

function [genderWords, genderVectors] = getGenderSpecific(filename,wordvecs_norm,word2Index,words_part)
fname = 'gender_specific.json'; 
fid = fopen(fname); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 
spec = jsondecode(str);

genderVectors = [];
counter2 = 0;
genderWords = lower(string(spec));
arr2 = ismember(genderWords,words_part);
for n = 1:length(genderWords)
    if arr2(n,1) == 1
        counter2 = counter2 + 1;
        genderVectors(counter2,:) = wordvecs_norm(word2Index(genderWords{n,1}),:);
    end
end
