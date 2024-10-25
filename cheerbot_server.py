import random
import datetime
from collections import defaultdict
from transformers import pipeline
import numpy as np

# Class to track user sentiment over time and profile users
class EmotionalStateTracker:
    def __init__(self):
        self.user_history = defaultdict(list)
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def add_entry(self, user_id, text):
        sentiment = self.analyze_sentiment(text)
        self.user_history[user_id].append({
            'timestamp': datetime.datetime.now(),
            'text': text,
            'sentiment': sentiment
        })
        return sentiment

    def analyze_sentiment(self, text):
        """Uses a pre-trained Hugging Face sentiment model to classify the input text."""
        result = self.sentiment_analyzer(text)[0]
        return result['label'].lower()  # 'positive' or 'negative'

    def get_emotional_state(self, user_id):
        """Return the average emotional state based on recent interactions."""
        history = self.user_history[user_id]
        if not history:
            return 'neutral'

        # Get the last 10 sentiment entries
        recent_sentiments = [entry['sentiment'] for entry in history[-10:]]
        positive_count = recent_sentiments.count('positive')
        negative_count = recent_sentiments.count('negative')

        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'

    def emotional_trend(self, user_id):
        """Tracks emotional shifts over time using moving averages."""
        history = self.user_history[user_id]
        if len(history) < 5:
            return 'not enough data'

        # Use moving average to calculate trends
        sentiments = [1 if entry['sentiment'] == 'positive' else -1 for entry in history]
        moving_avg = np.convolve(sentiments, np.ones(5)/5, mode='valid')

        if moving_avg[-1] > 0.5:
            return 'improving'
        elif moving_avg[-1] < -0.5:
            return 'worsening'
        else:
            return 'stable'

# CheerBot class to handle responses and advice generation
class CheerBot:
    def __init__(self):
        self.tracker = EmotionalStateTracker()

    def generate_advice(self, sentiment, trend):
        advice_pool = {
            'positive': [
                "Keep up the great work! You're on the right track.",
                "Your positive energy is inspiring, keep it going!",
                "Stay motivated and continue with your positivity!"
            ],
            'negative': [
                "It’s okay to have tough days. Practice self-care.",
                "Reach out for support if you're feeling down.",
                "Things will get better. Hang in there!"
            ],
            'neutral': [
                "Balance your emotions and stay mindful.",
                "Keep calm and continue practicing self-awareness.",
                "Neutral emotions are a good base to build on."
            ],
            'worsening': [
                "It seems things are getting harder. Try to take a break.",
                "If you're feeling overwhelmed, talk to someone you trust.",
                "Make time for things that make you happy and relaxed."
            ],
            'improving': [
                "You're making great progress! Keep up the self-care.",
                "You're turning things around; stay strong!",
                "Great job! Keep focusing on what’s working for you."
            ]
        }

        if trend == 'improving':
            return random.choice(advice_pool['improving'])
        elif trend == 'worsening':
            return random.choice(advice_pool['worsening'])
        else:
            return random.choice(advice_pool[sentiment])

    def process_user_input(self, user_id, user_input):
        # Analyze user sentiment
        sentiment = self.tracker.add_entry(user_id, user_input)
        trend = self.tracker.emotional_trend(user_id)

        # Generate personalized advice based on sentiment and emotional trend
        advice = self.generate_advice(sentiment, trend)
        return sentiment, trend, advice

# Simulate CheerBot interactions
if __name__ == "__main__":
    bot = CheerBot()
    user_id = "user123"

    while True:
        user_input = input("How are you feeling today? ")
        if user_input.lower() == 'exit':
            break

        sentiment, trend, advice = bot.process_user_input(user_id, user_input)
        print(f"Detected Sentiment: {sentiment}")
        print(f"Emotional Trend: {trend}")
        print(f"CheerBot says: {advice}")
