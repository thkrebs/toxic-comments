# Notebooks related to the Kaggle competition "Toxic Comments"

This project holds four notebooks which are related to the Kaggle Competition [https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge]

*Please note: *

1. that the repository does not hold the data provided in the context of the competition which would violate the Kaggle rules. Thus, if you want to run the code, you need to get data from the competition as well as the translated data used for test data augmentation
2. the kernels contained herein did not participate in the competition. Their purpose was to learn how to deal with this kind of problem


Fast Text embedding can be found here: [https://github.com/facebookresearch/fastText/blob/master/docs/english-vectors.md]
Glove embedding can be downloaded from here [https://nlp.stanford.edu/projects/glove/]
Translated files: [https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/48038#272289]

The repository holds the following notebook:

1. Baseline Model - toxic_comments_baseline.ipynb

   Using a 50 dimensional Glove vector
   
2. Model with 300 dimensional embedding - toxic_comments_crawl.ipynb

   Using a 300 dimensional Fasttext vector
   
3. Model with 300 dimensional embedding and augmented test data - toxic_comments_crawl-augmented_data.ipynb
