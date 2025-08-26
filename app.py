from flask import Flask, render_template, request, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import random
import os
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # render charts without a display
import matplotlib.pyplot as plt

# Ensure VADER lexicon is available
nltk.download('vader_lexicon')

app = Flask(__name__)
app.config['SECRET_KEY']  # Replace with a secure key in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///reviews.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


# -----------------------------
# Model for storing reviews
# -----------------------------
class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(1000), nullable=False)
    sentiment_score = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        short = (self.content or "")[:30].replace("\n", " ")
        return f"<Review {short}...>"


with app.app_context():
    db.create_all()


# -----------------------------
# Sentiment analysis
# -----------------------------
def analyze_sentiments(reviews):
    sid = SentimentIntensityAnalyzer()
    sentiments = [sid.polarity_scores(review) for review in reviews]
    return pd.DataFrame(sentiments)


# -----------------------------
# Extract products from reviews
# -----------------------------
def extract_products_from_reviews(reviews):
    """
    Keyword-based product extractor. Customize this list to your domain.
    """
    product_list = [
        "delivery", "packaging", "taste", "quality", "price", "refund",
        "app", "support", "burger", "pizza", "fries", "sauce", "combo",
        "subscription", "drink", "beverage"
    ]
    transactions = []
    for review in reviews:
        text = (review or "").lower()
        products_found = [p for p in product_list if p in text]
        if products_found:
            transactions.append(sorted(list(set(products_found))))
    return transactions


# -----------------------------
# Generate recommendations
# -----------------------------
def generate_recommendations(transactions, sentiments):
    if not transactions:
        return []

    # each row is a list of items -> one-hot basket
    transaction_df = pd.DataFrame(transactions)
    basket = pd.get_dummies(transaction_df.apply(pd.Series).stack()).groupby(level=0).sum()

    frequent_itemsets = apriori(basket, min_support=0.1, use_colnames=True)
    if frequent_itemsets.empty:
        return []

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    recommendations = []

    for idx, row in enumerate(transactions):
        if len(row) == 0:
            continue

        sentiment = sentiments.iloc[min(idx, len(sentiments) - 1)]
        positive_review = sentiment['compound'] > 0.05

        for _, rule in rules.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])

            if all(product in row for product in antecedents):
                if positive_review:
                    recommendations.append({
                        "rule": f"{', '.join(antecedents)} -> {', '.join(consequents)}",
                        "recommended_products": ', '.join(consequents),
                        "support": round(float(rule['support']), 3),
                        "confidence": round(float(rule['confidence']), 3),
                        "lift": round(float(rule['lift']), 3),
                        "sentiment": "Positive"
                    })
                else:
                    recommendations.append({
                        "rule": f"{', '.join(antecedents)} -> {', '.join(consequents)}",
                        "recommended_products": f"Consider improving {', '.join(consequents)}",
                        "support": round(float(rule['support']), 3),
                        "confidence": round(float(rule['confidence']), 3),
                        "lift": round(float(rule['lift']), 3),
                        "sentiment": "Negative"
                    })

    return recommendations


# -----------------------------
# Simple (keyword-based) recs
# -----------------------------
def generate_simple_recommendations(reviews):
    positive_keywords = [
        'good', 'like', 'love', 'wonderful', 'super', 'amazing',
        'marvellous', 'surprised', 'great', 'excellent', 'fantastic',
        'awesome', 'perfect'
    ]
    negative_keywords = [
        'hate', 'bad', 'poor', 'disgusting', 'waste', 'not happy',
        "don't like", 'disappointed', 'awful', 'not satisfied', 'horrible',
        'terrible', 'worst'
    ]
    neutral_keywords = [
        'okay', 'average', 'fine', 'moderate', 'fair', 'sufficient',
        'normal', 'acceptable', 'regular', 'standard', 'routine',
        'usual', 'ordinary', 'typical', 'not sure', 'perhaps', 'maybe',
        'decent', 'not bad', 'satisfactory'
    ]

    positive_recommendations = [
        "It's great to see your positive experience! We recommend trying our new flavors as well; they're fantastic!",
        "Since you enjoyed this product, you might also like our complementary items that enhance the experience!",
        "We're glad you loved this! You should check out our loyalty deals for even better value!",
        "Glad to hear you're satisfied! Don't miss our upcoming promotions that might interest you.",
        "Based on your positive review, you might enjoy our subscription service for regular deliveries!"
    ]

    negative_recommendations = [
        "We appreciate your feedback! Please reach out to our customer service for immediate assistance.",
        "Thanks for sharing your experience. Our troubleshooting guide might offer a quick solution.",
        "Sorry to hear that! Check our FAQs for tips that can help resolve it quickly.",
        "Thanks for your honesty! Stay tuned — we’re actively improving this area.",
        "We value your input. Our customer care can offer personalized support.",
        "We’re sorry about the experience. Please try our latest update — we’re continually improving."
    ]

    neutral_recommendations = [
        "Thanks for your balanced feedback! We’re working on making things even better.",
        "We appreciate your input. Do check back soon — improvements are always ongoing.",
        "Your thoughts are valuable. Stay tuned for upcoming updates that may enhance your experience.",
        "Thanks for the fair review! We'd love to hear more details to help us improve.",
        "We’re glad things were okay! We aim to make them great next time."
    ]

    selected_recommendations = []
    for review in reviews:
        content = (review or "").lower()
        if any(keyword in content for keyword in positive_keywords):
            recommendation = random.choice(positive_recommendations)
        elif any(keyword in content for keyword in negative_keywords):
            recommendation = random.choice(negative_recommendations)
        elif any(keyword in content for keyword in neutral_keywords):
            recommendation = random.choice(neutral_recommendations)
        else:
            recommendation = "Thank you for your feedback! We're always looking to improve."
        selected_recommendations.append(recommendation)

    return selected_recommendations


# -----------------------------
# Chart generation (bar + pie)
# -----------------------------
def ensure_graph_dir():
    graph_dir = os.path.join(app.root_path, "static", "images")
    os.makedirs(graph_dir, exist_ok=True)
    return graph_dir


def generate_charts():
    """
    Creates bar and pie charts of sentiment categories and saves as PNG.
    """
    reviews = Review.query.all()
    if not reviews:
        return None, None

    # Classify into Positive/Neutral/Negative
    def label(score):
        if score is None:
            return "Neutral"
        if score > 0.05:
            return "Positive"
        if score < -0.05:
            return "Negative"
        return "Neutral"

    labels = [label(r.sentiment_score) for r in reviews]
    counts = pd.Series(labels).value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)

    graph_dir = ensure_graph_dir()
    bar_path = os.path.join(graph_dir, "sentiment_bar.png")
    pie_path = os.path.join(graph_dir, "sentiment_pie.png")

    # Bar chart
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color=["#2ecc71", "#3498db", "#e74c3c"])  # Green, Blue, Red
    plt.title("Sentiment Counts")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150)
    plt.close()

    # Pie chart
    plt.figure(figsize=(5, 5))
    counts.plot(kind="pie", autopct="%1.1f%%", startangle=140, colors=["#2ecc71", "#3498db", "#e74c3c"])
    plt.title("Sentiment Distribution")
    plt.ylabel("")  # hide y-label
    plt.tight_layout()
    plt.savefig(pie_path, dpi=150)
    plt.close()

    # Return filenames relative to /static/images
    return "sentiment_bar.png", "sentiment_pie.png"

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def welcome():
    return render_template("welcome.html")


@app.route("/give_review", methods=["GET", "POST"])
def give_review():
    if request.method == "POST":
        review = (request.form.get("review") or "").strip()
        if review:
            sid = SentimentIntensityAnalyzer()
            sentiment = sid.polarity_scores(review)['compound']
            new_review = Review(content=review, sentiment_score=sentiment)
            db.session.add(new_review)
            db.session.commit()
            flash('Thank you for your feedback!', 'success')
            return render_template("index.html", sentiment=sentiment)
        else:
            flash("Please write a review before submitting.", "warning")
            return redirect(url_for("give_review"))
    return render_template("index.html")


@app.route('/get_recommendations')
def get_recommendations():
    reviews = [r.content for r in Review.query.all()]
    if not reviews:
        flash("Please submit at least one review to get recommendations.", "warning")
        return redirect(url_for("give_review"))

    sentiments = analyze_sentiments(reviews)
    transactions = extract_products_from_reviews(reviews)

    product_recommendations = generate_recommendations(transactions, sentiments)
    simple_recommendations = generate_simple_recommendations(reviews)

    return render_template(
        'recommendations.html',
        product_recommendations=product_recommendations,
        simple_recommendations=simple_recommendations
    )


@app.route("/stored_reviews")
def view_stored_reviews():
    stored_reviews = Review.query.order_by(Review.created_at.desc()).all()
    return render_template("stored_reviews.html", stored_reviews=stored_reviews)


@app.route("/charts")
def charts():
    bar_file, pie_file = generate_charts()
    if bar_file is None:
        flash("No data yet. Submit some reviews to see charts!", "info")
        return redirect(url_for("give_review"))
    return render_template("charts.html", bar_file=bar_file, pie_file=pie_file)


# -----------------------------
# Optional: Seed demo data route
# -----------------------------
@app.route("/seed_demo")
def seed_demo():
    demo_texts = [
        "Loved the burger and fries! Great taste and fast delivery.",
        "The delivery was late and packaging was bad. Not happy.",
        "Quality is amazing. The price is fair. Support team was helpful.",
        "I hate the new app design, it's confusing.",
        "Pizza was okay, sauce was great, combo is value for money.",
        "Refund process was smooth. Appreciate the quick response."
    ]
    sid = SentimentIntensityAnalyzer()
    for t in demo_texts:
        score = sid.polarity_scores(t)['compound']
        db.session.add(Review(content=t, sentiment_score=score))
    db.session.commit()
    flash("Seeded demo reviews.", "success")
    return redirect(url_for("view_stored_reviews"))


if __name__ == "__main__":
    # Ensure static/images exists for charts & backgrounds
    ensure_graph_dir()
    app.run(debug=True)

