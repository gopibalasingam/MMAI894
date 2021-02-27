# Overview
The objective of the project is to explore whether an NLP sentiment model can improve a deep learning stock price prediction model. The project's scope is to develop an NLP model that collects data from equity-oriented discussion groups on Reddit. The model will use this information to generate a sentiment score as an input into one or more deep learning stock price prediction model(s). The data for the deep learning model are historical and real-time financial information sourced from Yahoo Finance. The training, prediction, and analysis goal (i.e. prediction phase) is to evaluate if social media feeds improve forecasting accuracy of the model versus evaluating forecasting accuracy without social media feeds. 

# Design

### Sentimental Analysis
Sentiment analysis identifies whether a given statement for a stock is positive or negative sentiment. Our system uses this to get a set of Reddit data for a given timeseries range and determines a compound score for the positive or negative Reddit comments in the data set.

### Stock Analysis
The stock price analysis part helps collect the general trend of the stock prices for a given timeseries range. 

### Deep learning Model
The deep learning model is the backbone of the implemented system. It consists of a mixed input multi-branched deep learning neural network. The outputs of each branch are concatenated and correlated to obtain the output stock price and sentiment score (polarity).  The combined knowledge of both types of the data, the system is empowered to provide precise and consistent predictions on the stock prices.

### Implementation Architecture
![Design](https://github.com/gopibalasingam/stockprediction/blob/main/stockprediction.png)
