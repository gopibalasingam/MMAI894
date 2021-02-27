import pandas as pd
import datetime as dt
import pandas_datareader as pdr
import numpy as np
import praw
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from praw.models import MoreComments
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Initialize variables
# Input variables: 1) Stock symbol 2) Date range 3) Subreddit group 4) Training/Test data split
chatter = []
results = []
stock_symbol = 'TSLA'
stock_search = 'TSLA'
stock_date = '1-Jan-18'
stock_date_split = '2020-01-01'
reddit_group = 'Baystreetbets'
sentiment_analyzer = SentimentIntensityAnalyzer()

# Invoke Yahoo Finance API to get histrorical stock data
df_stock = pdr.get_data_yahoo(stock_symbol, stock_date) 

# Initialize Reddit API
reddit = praw.Reddit(client_id='joVnbeY7KKVvUw', 
                     client_secret='fAShoF-QCt7HaJDYYFuahLbY6ihtQw', 
                     user_agent='Gopi_Balasingam')

# Collect top comment data from subreddit group
for submission in reddit.subreddit(reddit_group).top(limit=1000):
    for comment in submission.comments:
        if isinstance(comment, MoreComments):
            continue
        if (re.search(stock_search, comment.body)):
            chatter.append([str(submission.title) + " " + comment.body, 
                            str(dt.datetime.fromtimestamp(submission.created)
                            .strftime('%Y-%m-%d'))])  
   
# Compute the sentiment score   
for line in chatter:
    pol_score = sentiment_analyzer.polarity_scores(str(line[0]))
    pol_score['Headline'] = str(line[0])
    pol_score['Date'] = str(line[1])
    results.append(pol_score)

df_sentiment = pd.DataFrame.from_records(results, index='Date')
df_sentiment.index = pd.to_datetime(df_sentiment.index)
df = pd.DataFrame.merge(df_stock, df_sentiment, on='Date')
df = df.reset_index()
df = df.loc[df.groupby('Date')['compound'].idxmax()]
df = df.reset_index()

# Build the Dataframe which will be the input into the models
df = df.append(df_stock.reset_index()).fillna(0)
df = df.reset_index()
df = df.loc[df.groupby('Date')['compound'].idxmax()]
df = df.rename(columns={'compound' : 'Polarity'})
df = df.drop(columns=['level_0', 'index', 'neg', 'neu', 'pos', 'Adj Close', 'Headline'])
df = df.set_index('Date', drop = True)
df.index = pd.to_datetime(df.index)

# Split training and test data sets based on date range
data_training = df.loc[(df.index <= stock_date_split)]
data_test = df.loc[(df.index >= stock_date_split)]

data_test = data_test.reset_index()
data_training = data_training.reset_index()

print(data_training.shape)
print(data_training)
print(data_test)

data_training = data_training.drop(['Date'], axis = 1)
data_training_test = data_training
data_test = data_test.drop(['Date'], axis = 1)

# The values in the training data are not in the same range
# For getting all the values in between the range 0 to 1 use MinMaxScalar() 
# to improves the accuracy of prediction
scaler = MinMaxScaler()
data_training = scaler.fit_transform(data_training)
print(data_training)

# This sections of the code divides the data into chunks of 60 rows corresponds to 
# the length of data_traning. After dividing we are converting X_train and y_train
# into numpy arrays
X_train = []
y_train = []

for i in range(60, data_training.shape[0]):
    X_train.append(data_training[i-60:i])
    y_train.append(data_training[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

print(X_train.shape)

# Build model
model = Sequential()

model.add(LSTM(units = 60, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 6)))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.summary()

model.compile(optimizer='adam', loss = 'mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)

# past_60_days contains the data of the past 60 days required to predict the opening of the 1st day in the test data set.
past_60_days = data_training_test.tail(60)
print(past_60_days)

# We are going to append data_test to past_60_days and ignore the index of data_test
df = past_60_days.append(data_test, ignore_index = True)
print(df)

# Similar to the training data set we have to scale the test data so that all the values are in the range 0 to 1.
inputs = scaler.fit_transform(df)
inputs

# Prepare the test data like the training data.
X_test = []
y_test = []

for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i])
    y_test.append(inputs[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
X_test.shape, y_test.shape

y_pred = model.predict(X_test)

print(scaler.scale_)

# Calcualte the original price back again
scale = 1/scaler.scale_[0]
print(scale)

# Prediction on the normal price scale
y_pred = y_pred*scale
y_test = y_test*scale

# Visualizing the results
plt.figure(figsize=(14,5))
plt.plot(y_test, color = 'red', label = 'Real Stock Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted Stock Price')
plt.title(stock_symbol + ' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(stock_symbol + ' Stock Price')
plt.legend()
plt.show()