## Patient Sentiment Analysis on Healthcare Services
**Project Overview** 

The objective of this project is to analyze patient reviews to assess the sentiment and identify areas for improvement in healthcare services. By applying Natural Language Processing (NLP) and machine learning techniques, we aim to provide actionable insights that healthcare providers can use to enhance patient satisfaction.

**Tools and Skills**

Programming Language: Python
Libraries: pandas, re, sklearn (TfidfVectorizer, LogisticRegression, MultinomialNB, SVC, RandomForestClassifier, classification_report)
Techniques: Natural Language Processing (NLP), Machine Learning, Data Visualization, SQL

**Key Components**

1. Data Collection
Sources: Patient reviews were collected from healthcare provider websites, review platforms, and patient surveys.
Storage: The collected reviews were stored in a CSV file.
2. Data Preprocessing
Text Cleaning: Convert all text entries to strings, remove special characters and digits, convert to lower case, and remove leading and trailing whitespaces.
Transformation: Use TF-IDF (Term Frequency-Inverse Document Frequency) to convert text data into numerical features suitable for machine learning models.
3. Sentiment Labeling
Categories: Based on the rating provided in the reviews, the sentiments were categorized as:
Positive (ratings 4 and 5)
Neutral (rating 3)
Negative (ratings 1 and 2)
4. Model Training and Evaluation
Models Used: Logistic Regression, Naive Bayes, Support Vector Machine (SVM), and Random Forest.
Training: Each model was trained on the TF-IDF features derived from the preprocessed text.
Evaluation: Model performance was evaluated using metrics such as precision, recall, and F1-score.

**Model Evaluation Results** 

**Naive Bayes:**

Accuracy: 42%  
Positive Sentiment F1-Score: 52%  

**Support Vector Machine:**

Accuracy: 41%  
Positive Sentiment F1-Score: 50%  

**Random Forest:**

Accuracy: 41%  
Positive Sentiment F1-Score: 50%  

**Logistic Regression:**

Accuracy: 41%  
Positive Sentiment F1-Score: 50%  

**Insights and Recommendations
Model Performance:**

Naive Bayes achieved the highest accuracy and F1-score for positive sentiment, making it the best performing model among those tested.
All models struggled to predict neutral sentiments, indicating a need for better representation and training data for this category.

**Data Augmentation:**

Increase the dataset size, especially for neutral reviews, to provide more balanced training data.
Collect more labeled data to improve model training.

**Advanced Models:**

Explore more sophisticated models such as BERT or other transformer-based models that might capture nuances better than traditional classifiers.

**Feature Engineering:**

Incorporate additional text processing techniques, such as n-grams (bigrams or trigrams) and word embeddings.
Experiment with different feature extraction methods and hyperparameter tuning.

**Continuous Improvement:**

Implement a feedback loop to continuously collect new patient reviews and update the analysis.
Regularly refine and retrain models with new data to improve accuracy and relevance.

**Conclusion**
This sentiment analysis project provides a foundation for understanding patient feedback on healthcare services. Although the models showed moderate success in predicting sentiments, there is significant room for improvement. By following the recommendations, the analysis can be refined to provide more accurate and actionable insights, ultimately aiding healthcare providers in enhancing their services.
