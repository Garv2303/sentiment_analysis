## Imports

import pandas as pd
import csv
import nltk
import os.path as checkcsv

## Downloads

def sepposnegcom(comment_file):

    ## Reading Dataset

    dataset = pd.read_csv(comment_file, encoding_errors = 'ignore')
    dataset = dataset.iloc[:, 0:]

    ## Getting Full Comments to csv file

    # full_com = dataset
    # full_comments = full_com.to_csv("Full Comments.csv")

    ## Sentiment analysis of comments using vadar sentiment analyser

    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    analyser = SentimentIntensityAnalyzer()

    def vader_sentiment_result(sent):
        scores = analyser.polarity_scores(sent)
        if scores["compound"] >= 0.05:
            return 1  # Positive
        elif scores["compound"] <= -0.05:
            return 0  # Negative
        else:
            return 2  # Neutral (if you want to categorize neutral comments)


    dataset['vader_sentiment'] = dataset['Comment'].apply(lambda x : vader_sentiment_result(x))
    print(dataset[['Comment', 'vader_sentiment']])  # Debugging output to check the classification


    

    ## Separating Positive and Negative Comments

    # Separate comments based on vader_sentiment and save them in corresponding CSV files
    pos_comments = dataset[dataset['vader_sentiment'] == 1]
    neg_comments = dataset[dataset['vader_sentiment'] == 0]

# Save positive and negative comments to CSV
    pos_comments.to_csv('1.csv', index=False)
    neg_comments.to_csv('0.csv', index=False)

# Debugging: print confirmation
    print(f"Positive Comments saved to 1.csv, count: {len(pos_comments)}")
    print(f"Negative Comments saved to 0.csv, count: {len(neg_comments)}")

    pos_comments = pd.read_csv('1.csv')
    neg_comments = pd.read_csv('0.csv')
    print("Positive Comments:", pos_comments)
    print("Negative Comments:", neg_comments)

    
    if checkcsv.exists('1.csv') == False:                             # If 1.csv file does not exist, it creates one empty 1.csv file.
        with open('1.csv', 'w', encoding='UTF8', newline='') as f1:
            writer1 = csv.writer(f1)
            header1 = ['Empty', 'Empty', 'Empty']
            row1 = ['No Positive Comments', 'No Positive Comments', 'No Positive Comments']
            writer1.writerow(header1)
            writer1.writerow(row1)

    if checkcsv.exists('0.csv') == False:                             # If 1.csv file does not exist, it creates one empty 1.csv file.
        with open('0.csv', 'w',encoding='UTF8', newline='') as f0:
            writer0 = csv.writer(f0)
            header0 = ['Empty', 'Empty', 'Empty']
            row0 = ['No Negative Comments', 'No Negative Comments', 'No Negative Comments']
            writer0.writerow(header0)
            writer0.writerow(row0)
    
    pos = (pd.read_csv("1.csv", engine = 'python')).iloc[:, :-1]
    neg = (pd.read_csv("0.csv", engine = 'python')).iloc[:, :-1]

    positive_comments = pos.to_csv("Positive Comments.csv", index=False)
    negative_comments = neg.to_csv("Negative Comments.csv",index=False)

    video_positive_comments = str(len(pos.axes[0])) + ' Comments'  #Finding total rows in positive comments
    video_negative_comments = str(len(neg.axes[0])) + ' Comments'  #Finding total rows in negative comments
    
    if (pd.read_csv('1.csv', nrows=0).columns.tolist())[0] == 'Empty':
        video_positive_comments = '0 Comments'
    if (pd.read_csv('0.csv', nrows=0).columns.tolist())[0] == 'Empty':
        video_negative_comments = '0 Comments'

    ## return function
    return positive_comments, negative_comments, video_positive_comments, video_negative_comments
