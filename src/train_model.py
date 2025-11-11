from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pandas as pd
import joblib

df = pd.read_csv('data/products.csv')

df.columns = df.columns.str.replace(' ', '').str.replace('_', '')
df = df.drop(columns=['productID', 'ProductCode', 'NumberofViews', 'MerchantRating', 'ListingDate'])

df = df.dropna()

replace_map = {
    'Fridge Freezers': 'Fridge',
    'Fridges': 'Fridge',
    'Freezers': 'Fridge',
    'fridge': 'Fridge',

    'Washing Machines': 'Washing Machine',

    'Mobile Phones': 'Mobile Phone',
    'Mobile Phone': 'Mobile Phone',

    'CPUs': 'CPU',
    'CPU': 'CPU',

    'TVs': 'TV',

    'Dishwashers': 'Dishwasher',

    'Digital Cameras': 'Digital Camera',

    'Microwaves': 'Microwave',
}
df['CategoryLabel'] = df['CategoryLabel'].replace(replace_map)

x = df['ProductTitle']
y = df['CategoryLabel']

vectorizer = TfidfVectorizer()
x_vec = vectorizer.fit_transform(x)

model = LinearSVC()
model.fit(x_vec, y)

joblib.dump(model, 'model/category_prediction_model.pkl')
joblib.dump(vectorizer, 'model/tfidf.pkl')
print("Model trained and saved!")



