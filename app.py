from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]
    data = vectorizer.transform([news])
    prediction = model.predict(data)

    result = "🌐 This News is REAL" if prediction[0] == 1 else "🚨 This News is FAKE"
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
