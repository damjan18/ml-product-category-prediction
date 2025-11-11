import joblib

# Učitaj model i vektorizator
model = joblib.load('model/category_prediction_model.pkl')
vectorizer = joblib.load('model/tfidf.pkl')

print("Model and vectorizer loaded successfully!")
print("Type 'exit' at any point to stop.\n")

while True:
    title = input("Enter product title: ")
    if title.lower() == "exit":
        print("Exiting...")
        break

    # Transformiši unos korisnika pomoću TF-IDF vektorizatora
    X = vectorizer.transform([title])

    # Napravi predikciju
    prediction = model.predict(X)[0]
    print(f"Predicted category: {prediction}\n" + "-" * 40)