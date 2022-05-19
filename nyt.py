import json
import requests
import nltk
import sklearn
import numpy as np
import pandas as pd
import plotly.express as px

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix

nltk.download('vader_lexicon')

def load_nyt_most_popular(key):

    """
    Loads in most popular NYT articles from last 30 days 
    Args:
    - str: NYT API key
    Returns:
    - tuple: list of most popular article headlines, oldest article date
    """

    headers = {
    "user-agent": "CIS 192 Fall 2021 Final Proj by ashahn"
    }

    article_heds = []
    article_dates = []

    mostpop_url = "https://api.nytimes.com/svc/mostpopular/v2/viewed/30.json?api-key={}"
    mostpop_url = mostpop_url.format(key)

    response = requests.get(mostpop_url, headers=headers)
    print(response)
    json_output = response.json()
    #print response in more readable format
    print(json.dumps(json_output, indent = 4))

    articles = json_output["results"]

    for x in articles:
        article_heds.append(x["title"])
        article_dates.append(x["updated"])
        #published_date might include really old dates from articles years ago that for some reason are popular now again
        #not helpful when needing to determine which recent articles to include from archive API so use last updated date 

    oldest = find_oldest_date(article_dates)
    return article_heds, oldest

def find_oldest_date(dates):

    """
    Finds the oldest date from a given list 
    Args:
    - list of str: dates representing when articles were published
    Returns:
    - str: oldest date in list
    """

    oldest = "0000000000"
    for date in dates:
        #check if date is older than oldest
        if int(date[2:4]) < int(oldest[2:4]): #if older year
            oldest = date
        elif int(date[2:4]) == int(oldest[2:4]): #if same year
            if int(date[5:7]) < int(oldest[5:7]): #if older month
                oldest = date
            elif int(date[5:7]) == int(oldest[5:7]): #if same month
                if int(date[8:]) < int(oldest[8:]): #if older day
                    oldest = date
    return oldest

def load_nyt_archive(key, month1, month2, year):

    """
    Loads in NYT articles from specified months (should be the previous and current month so
    timeframe matches most popular API) and year (should be current year)
    Args:
    - str: NYT API key
    - int: previous month (xx)
    - int: current month (xx)
    - int: current year (xxxx)
    Returns:
    - tuple: list of unpopular article headlines, list of popular headlines                               
    """

    pop_heds, oldest = load_nyt_most_popular(key)

    headers = {
    "user-agent": "CIS 192 Fall 2021 Final Proj by ashahn"
    }
    #load articles from month 1
    article_heds1 = []
    archive_url = "https://api.nytimes.com/svc/archive/v1/{}/{}.json?api-key={}"
    archive_url = archive_url.format(year, month1, key)

    response = requests.get(archive_url, headers=headers)
    json_output = response.json()
    articles = json_output["response"]["docs"]

    counter = 0
    for x in articles:
        #only add 350 archived articles (signficantly more than this slows down terminal)
        if counter == 350:
            break
        date = x["pub_date"][:10] 
        #Only add articles that were published in last 30 days 
        if is_within_timeframe(date, oldest):
            article_heds1.append(x["headline"]["main"])
        counter += 1
    #load articles from month 2
    article_heds2 = []
    archive_url = "https://api.nytimes.com/svc/archive/v1/{}/{}.json?api-key={}"
    archive_url = archive_url.format(year, month2, key)

    response = requests.get(archive_url, headers=headers)
    json_output = response.json()
    articles = json_output["response"]["docs"]

    counter = 0
    for x in articles:
        if counter == 350:
            break
        #Don't need to check if article is within timeframe, because month2 is
        #the current month so guaranteed to be published in last 30 days
        article_heds2.append(x["headline"]["main"]) 
        counter += 1
    article_heds1.extend(article_heds2)
    #delete articles that are deemed "most popular"
    for x in article_heds1:
        for y in pop_heds:
            if x == y:
                article_heds1.remove(x)
    
    return (article_heds1, pop_heds)

def is_within_timeframe(date, oldest):

    """
    Checks if the given date is equal to or more recent than oldest date in most popular API
    Args:
    - str: date to check 
    - str: oldest date in most popular API
    Returns:
    - bool: if date is more recent or equal to the oldest date
    """

    #check if date is older than oldest
    if int(date[2:4]) < int(oldest[2:4]): #if older year
        return False
    elif int(date[2:4]) == int(oldest[2:4]): #if same year
        if int(date[5:7]) < int(oldest[5:7]): #if older month
            return False
        elif int(date[5:7]) == int(oldest[5:7]): #if same month
            if int(date[8:]) < int(oldest[8:]): #if older day
                return False
    return True

def bag_of_word(heds, popheds):

    """
    Turns headlines into matrix of token counts which is used to train a model to predict if articles based on 
    headlines will be popular 
    Args:
    - list of str: unpopular article headlines
    - list of str: popular article headlines
    Returns:
    - None
    - Prints accuracy and confusion matrix
    """

    #score of 0 is not popular
    #score of 1 is popular
    y_heds = [0]*len(heds)
    x_heds_train, x_heds_test, y_heds_train, y_heds_test = train_test_split(heds, y_heds, test_size=0.2, random_state=42)

    y_popheds = [1]*len(popheds)
    x_popheds_train, x_popheds_test, y_popheds_train, y_popheds_test = train_test_split(popheds, y_popheds, test_size=0.2, random_state=42)

    model = CountVectorizer()
    train_vec = model.fit_transform(x_heds_train + x_popheds_train).todense()
    vocab = []
    for word, idx in model.vocabulary_.items():
        vocab.append(word)

    model = CountVectorizer(vocabulary = vocab)
    test_vec = model.transform(x_heds_test + x_popheds_test).todense()

    nb = GaussianNB()
    nb.fit(train_vec.tolist(), y_heds_train + y_popheds_train)
    y_pred = nb.predict(test_vec.tolist())
    accuracy = accuracy_score(y_heds_test + y_popheds_test, y_pred)

    print("Accuracy: ", accuracy)

def get_polarity(heds, popheds):
    
    """
    Computes the polarity scores of the unpopular and popular article headlines and shows visualization of findings
    Args:
    - list of str: unpopular article headlines
    - list of str: popular article headlines
    Returns:
    - None
    - Shows visualization of all articles' polarity scores
    - Shows visualization comparing just the negative percentage polarity scores of popular and unpopular articles 
    """

    hed_polarity_scores = []
    pophed_polarity_scores = []

    for hed in heds:
        sid = SentimentIntensityAnalyzer()
        hed_polarity_scores.append(sid.polarity_scores(hed))
    #create dataframe and append unpopular headlines' polarity scores
    hed_df = pd.DataFrame(columns = ["neg", "neu", "pos", "compound"])
    hed_df = hed_df.append(hed_polarity_scores, ignore_index = True, sort = False)
    hed_df["headline"] = heds
    hed_df["Popularity"] = ["Not popular"]*len(heds)

    for pophed in popheds:
        sid = SentimentIntensityAnalyzer()
        pophed_polarity_scores.append(sid.polarity_scores(pophed))
    #create dataframe and append popular headlines' polarity scores
    pophed_df = pd.DataFrame(columns = ["neg", "neu", "pos", "compound"])
    pophed_df = pophed_df.append(pophed_polarity_scores, ignore_index = True, sort = False)
    pophed_df["headline"] = popheds
    pophed_df["Popularity"] = ["Popular"]*len(popheds)
    
    #combine unpopular and popular headline dataframes
    combined_df = hed_df.append(pophed_df)

    #create plotly ternary scatter plot to compare all polarity scores
    fig = px.scatter_ternary(combined_df, a="neu", b="neg", c="pos", hover_name="headline", color = "Popularity",
                             title = "Polarity percentage for NYT headlines")
    fig.update_layout(title_x=0.5, title_font = dict(size=25))
    fig.show()

    #create plotly bar graph to compare just negative polarity scores
    fig1 = px.bar(x=["Not popular", "Popular"], y=[np.mean(hed_df["compound"]), np.mean(pophed_df["compound"])],
                  title = "Negative polarity percentage for NYT headlines")
    fig1.update_layout(title_x=0.5, title_font = dict(size=25))
    fig1.update_xaxes(title = "Popularity of the article", title_font=dict(size=16))
    fig1.update_yaxes(title = "Neg polarity percentage", title_font=dict(size=16))
    fig1.show()

    
class Article:

    def __init__(self, headline):

        """
        Construct an instance of the Article class
        Attributes: 
        - self.headline: str representing the article's headline
        - self.polarityscore: float representing the article's polarity score based on its headline
        Args:
        - headline (str)
        """

        if not isinstance(headline, str):
            raise Exception("Headline is not a string.")

        self.headline = headline
        self.polarityscore = 0.0
    
    def __str__(self):

        """
        Defines the string representation of a given article instance
        Args:
        - None
        Returns:
        - str: article's headline
        """
        
        return(self.headline)

    def add_polarity_score(self):

        """
        Gets compounded polarity score for article's headline
        Args:
        - None
        Returns:
        - float: polarity score
        """
        sid = SentimentIntensityAnalyzer()
        self.polarityscore = sid.polarity_scores(self.headline)["compound"]
        return self.polarityscore


def main():
    """main function"""
    key = "071M3wdZUfQiyAJTwuNO6TftiP0BMbUh"
    heds, popheds = load_nyt_archive(key, 4, 5, 2022)
    bag_of_word(heds, popheds)
    get_polarity(heds, popheds)
    
    
    #Testing Article class 
    my_Article = Article("Tracking the Suspect in the Fatal Kenosha Shootings")
    print(my_Article)
    my_Article.add_polarity_score()
    print(my_Article.polarityscore)
    

if __name__ == '__main__':
    """
    Test implementation here by running python3 nyt.py in terminal
    """
    main()