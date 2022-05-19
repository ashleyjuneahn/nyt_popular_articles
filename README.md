CIS 192 Final Project 
ashahn

My project 1) predicts whether an article will be 'popular' or 'unpopular' given its headline and 2) examines the polarity of popuar and unpopular article headlines to see if there is any significant difference between the two. The expectation is that popular headlines contain more polarizing language. I use the requests module to retrieve the JSON response from NYT APIs which contain information on article headlines and when articles were published. For the first part of my project, I use the Gaussian Naives Bayes Classifier to train a set of the popular and unpopular headlines and use the sklearn.metrics module to show the model's accuracy and confusion matrix. For the second part of my project, I use NLTK's Sentiment Analysis using VADER (Valence Aware Dictionary for Sentiment Reasoning) to score polarity of headlines. I then put the headlines' polarity scores into a pandas dataframe and use plotly visualization to show the results.

I also include an Article class to easily obtain the compound polarity score for an article with a specific headline. This is useful to check the visualizations.
Helper methods are also included to assist with determining which articles from the achived articles API matches the timeframe of the most popular articles API (which only includes articles from the last 30 days). 

A NYT API key is needed in order to access the API. The key I use is included in the main function.
There is an API call limit at 4,000 requests per day and 10 requests per minute.

Modules:
- json
- requests
- nltk
- pandas
- numpy
- plotly
- sklearn