# jewelry_review_dataset_analysis

The repository consists of analysis of Amazon reviews dataset which contains real reviews for jewelry products sold on Amazon. 

Data cleaning steps 
- convert the all reviews into the lower case.
- remove the HTML and URLs from the reviews
- remove accents
- remove non-alphabetical characters
- remove extra spaces
- perform contractions on the reviews, e.g., won’t → will not.
- spell correction
- emoji to word conversion 

Feature Extraction
- TF-IDF
- Word2Vec

Models tried
- Perceptron
- SVM
- Logistic Regression
- Multinomial Naive Bayes
- FNN with weighted features
- FNN with concatenated features
- RNN
- GRU

