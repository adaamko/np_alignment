## The dictionary baseline
- The baseline uses bilingual dictionaries. Precisely the merged version of Hokoto and a dictionary made with Wikt2dict.
- First the method makes a lookup in the dictionary to find out the hungarian translation of the words for the english NP.
- After, if the translation words corresponds to the hungarian NP to any amount, we add an edge between them (so we don't search for a maximum value, we say if two NP has a common word, we add an edge)
- We do lemmatization and stopword filtering on both sides. For english we used \texttt{spacy}, for hungarian \texttt{emmorph}.
- We keep the stopwords if there are only stopwords in the NP:
- We calculate Levhenstein distance as well, so words like Ocania -> Óceánia can make a pair. 
- On the hungarian side we also added morphological analysis of the words (e.g. Gondolatrendőrség = gondolat+rendőrség[/N])


## The embedding baseline
- Aside from the baseline built on dictionary pairs, we also experimented with a baseline completely built on embeddings.
- We used the cross-lingual Fasttext embeddings(MUSE https://github.com/facebookresearch/MUSE)
- For each NP pair we also used the previously described filtering method (lemmatization, stopword filtering)
- We calculate cosine similarity for each word pair from the NPs
- We take the maximum number from the similarities
- We calculated the optimum threshold after we consider an NP pair aligned
- This threshold was set based on the training data
- The threshold is 0,5

## The Deep learning method
- We experimented with building a baseline using BERT 
- We used the bert-base-multilingual-cased pretrained model of BERT
- We used the pretrained model to obtain sentence embeddings for each sentence pair
- This was done by concatenating an english and a hungarian sentence together separated by the [SEP] keyword
- After running through the sentences on BERT, we obtained the embeddings from the weights of the last hidden layers
- The embeddings were generated in the context of the sentences, and then the embeddings of the words of the NP-s were cut from the sentence embeddings with the corresponding indices
- The architecture of the network consists of an LSTM layer working with the BERT word embeddings generating a feature vector for each NP pair. Then we define a linear layer to reduce the vector and get the probabilites of each label
- The method is basically a binary classification of the NP pairs
- For the loss function we used Negative log likelihood loss
- We set the starting learning rate to 0.01 with Adam optimizer
- We used early stopping to avoid overfitting and mini batches
- Approximately there are six times more negative samples in the data than there are positive ones resulting in imbalanced classes
- First we dealt with the imbalanced classes with undersampling of the bigger class, but it didn't yield good results
- We also experimented with weighting the loss function with the inverse distribution of the number of the classes (best dev fscore:0.5822344587740906 precision 0.5345928685470995 recall 0.6391982182628062)
- Oversampling of the smaller class yielded the best results(dev prec:0.670596 dev rec:0.771953 dev fscore:0.717714)
- We used a weighted sampler which drawns from the smaller class with higher probability resulting in balanced data
- This way we didn't lose information, which was the case with undersampling of the bigger class
 