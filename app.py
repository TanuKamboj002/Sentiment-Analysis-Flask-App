from flask import Flask, render_template, request
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# ✅ Ensure vader_lexicon is downloaded
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

# Initialize analyzer
sia = SentimentIntensityAnalyzer()

import os
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from nltk.sentiment import SentimentIntensityAnalyzer

# Create Flask app
app = Flask(__name__)

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment_label = None
    sentiment_color = None
    scores = None
    text1 = ""

    if request.method == "POST":
        text1 = request.form["text1"]

        # Get sentiment scores
        scores = sia.polarity_scores(text1)

        # Determine label & color
        compound = scores["compound"]
        if compound >= 0.05:
            sentiment_label = "Positive 😊"
            sentiment_color = "positive"
        elif compound <= -0.05:
            sentiment_label = "Negative 😞"
            sentiment_color = "negative"
        else:
            sentiment_label = "Neutral 😐"
            sentiment_color = "neutral"

        # Save sentiment distribution chart
        labels = ["Positive", "Neutral", "Negative"]
        values = [scores["pos"], scores["neu"], scores["neg"]]

        plt.figure(figsize=(4, 4))
        plt.bar(labels, values, color=["green", "gray", "red"])
        plt.title("Sentiment Distribution")
        chart_path = os.path.join("static", "sentiment_chart.png")
        plt.savefig(chart_path)
        plt.close()

    return render_template(
        "form.html",
        sentiment_label=sentiment_label,
        sentiment_color=sentiment_color,
        scores=scores,
        text1=text1,
    )

if __name__ == "__main__":
    app.run(debug=True)
