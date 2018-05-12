%{
    EC503 - Learning from Data
    March 2018
    Word Embeddings De-biasing
    Function for Finding Vectors of Type passed
    Worked on by: Nidhi Tiwari & Frank Tranghese
%}

function [occupationWords, occupationVectors] = getVectorsOfType(filename,wordvecs_norm,word2Index,words_part)

fname = filename; 
fid = fopen(fname); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 
val = jsondecode(str);

occupationWords = {};
occupationVectors = [];
for k = 1:length(val)
    occupationWords{k,1} = char(val{k,1}(1,1));
end
% y = keys(word2Index);
arr = ismember(occupationWords,words_part);
counter = 0;
for m = 1:length(arr)   
    if arr(m,1) == 1
        counter = counter + 1;
        occupationVectors(counter,:) = wordvecs_norm(word2Index(occupationWords{m,1}),:);
    end
end