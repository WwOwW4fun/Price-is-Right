import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import CountVectorizer
from pricer.evaluator import evaluate
from pricer.items import Item
#import xgboost as xgb
train, val, test = Item.from_hub("lesserafimlover/items")
#Random pricer baseline
def random_pricer(item):
    return random.randrange(1,1000)

#Constant pricer

training_prices = [item.price for item in train]
training_average = sum(training_prices) / len(training_prices)
def constant_pricer(item):
    return training_average

#Natural Language Linear Regression

prices = np.array([float(item.price) for item in train])
documents = [item.summary for item in train]
np.random.seed(42)
vectorizer = CountVectorizer(max_features=2000, stop_words='english')
X = vectorizer.fit_transform(documents)
regressor = LinearRegression()
regressor.fit(X, prices)
def natural_language_linear_regression_pricer(item):
    x = vectorizer.transform([item.summary])
    return max(regressor.predict(x)[0], 0)

#Random Forest ML model

subset = 15_000
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=4)
rf_model.fit(X[:subset], prices[:subset])
def random_forest(item):
    x = vectorizer.transform([item.summary])
    return max(0, rf_model.predict(x)[0])


if __name__ == "__main__":
    evaluate(random_forest,test)