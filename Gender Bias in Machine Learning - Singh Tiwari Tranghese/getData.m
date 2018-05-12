%{
    EC503 - Learning from Data
    March 2018
    Word Embeddings De-biasing
    Function for importing data
    Data from Tolga Bolukbasi
    (https://drive.google.com/drive/folders/0B5vZVlu2WoS5dkRFY19YUXVIU2M?usp=sharing)
    Worked on by: Frank Tranghese
%}

function [words,wordVector] = getData(filename)

fileID = fopen(filename);
fmt = ['%s ' repmat('%f ',1,300)];
words = textscan(fileID,fmt);
fclose('all');

wordVector = cell2mat(words(:,2:end));
words = words(:,1);
end