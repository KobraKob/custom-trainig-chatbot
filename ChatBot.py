import nltk
import random
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Expanded training data for social media marketing
training_data = [
    # Existing base intents
    {"intent": "greeting", "patterns": ["hello", "hi", "hey", "howdy"],
     "response": ["Hello!", "Hi there!", "Hey! How can I help you with social media marketing?"]},
    {"intent": "goodbye", "patterns": ["bye", "goodbye", "see you"],
     "response": ["Goodbye!", "Thanks for chatting about social media marketing!", "Take care!"]},
    {"intent": "thanks", "patterns": ["thank you", "thanks", "appreciate it"],
     "response": ["You're welcome!", "Happy to help with your social media strategy!", "No problem!"]},
    {"intent": "how_are_you", "patterns": ["how are you", "how's it going?", "how do you do?"],
     "response": ["I'm great and ready to discuss social media marketing!",
                  "Doing well and excited to help you boost your online presence!"]},
    {"intent": "age", "patterns": ["how old are you", "what is your age", "how many years old"],
     "response": ["I'm a digital marketing assistant, focused on helping you succeed online!",
                  "Age doesn't matter - I'm here to support your social media goals!"]},

    # Social Media Marketing Specific Intents
    {"intent": "social_media_platforms", "patterns": [
        "which social media platforms should I use",
        "what platforms are best for marketing",
        "where should I focus my social media efforts"
    ], "response": [
        "The best platforms depend on your target audience. Facebook, Instagram, LinkedIn, Twitter, and TikTok are popular choices. Each platform has unique strengths for different business types.",
        "Consider your target demographic: LinkedIn for B2B, Instagram for visual brands, TikTok for younger audiences, Facebook for broad reach, and Twitter for real-time engagement.",
        "Your platform choice should align with where your target audience spends most of their time online."
    ]},

    {"intent": "content_strategy", "patterns": [
        "how do I create a social media content strategy",
        "what should I post on social media",
        "content ideas for social media"
    ], "response": [
        "Develop a content strategy by understanding your audience, creating a content calendar, mixing content types (educational, entertaining, promotional), and maintaining a consistent brand voice.",
        "Focus on creating valuable content: how-to guides, behind-the-scenes, user-generated content, testimonials, and interactive posts like polls or Q&As.",
        "Aim for a 80/20 rule: 80% valuable, entertaining content and 20% promotional content to keep your audience engaged."
    ]},

    {"intent": "posting_frequency", "patterns": [
        "how often should I post",
        "posting frequency on social media",
        "how many times a day should I post"
    ], "response": [
        "Posting frequency varies by platform: Instagram 1-2 times daily, Facebook 3-5 times weekly, Twitter 3-5 times daily, LinkedIn 1-2 times daily.",
        "Quality trumps quantity. It's better to post less frequently with high-quality, engaging content than to post constantly with low-value content.",
        "Use analytics to find your optimal posting times and frequency by tracking engagement rates."
    ]},

    {"intent": "hashtag_strategy", "patterns": [
        "how to use hashtags",
        "hashtag strategy",
        "best practices for hashtags"
    ], "response": [
        "Use a mix of popular and niche hashtags. Research relevant hashtags in your industry, create branded hashtags, and limit to 3-5 hashtags per post.",
        "Hashtags increase discoverability. On Instagram, use up to 10-15 relevant hashtags. On Twitter, stick to 1-2 hashtags per tweet.",
        "Create unique branded hashtags for campaigns and encourage user-generated content."
    ]},

    {"intent": "engagement_boost", "patterns": [
        "how to increase social media engagement",
        "get more likes and comments",
        "improve social media interaction"
    ], "response": [
        "Boost engagement by responding to comments, asking questions, running contests, using interactive features like polls, and creating shareable content.",
        "Engage with your audience authentically. Respond to comments, share user-generated content, and create content that encourages interaction.",
        "Use storytelling, behind-the-scenes content, and personal touches to make your brand more relatable."
    ]},

    {"intent": "advertising", "patterns": [
        "social media advertising",
        "how to run ads",
        "paid social media marketing"
    ], "response": [
        "Start with clear objectives: brand awareness, lead generation, or sales. Use platform-specific ad tools with precise targeting options.",
        "Begin with a small budget, A/B test your ads, use compelling visuals, and create targeted campaigns for specific audience segments.",
        "Facebook Ads and Instagram Ads offer robust targeting. LinkedIn is great for B2B, while TikTok Ads work well for younger demographics."
    ]},

    {"intent": "analytics", "patterns": [
        "how to track social media performance",
        "social media analytics",
        "measuring social media success"
    ], "response": [
        "Use platform-specific analytics tools like Facebook Insights, Instagram Insights, and third-party tools like Hootsuite or Sprout Social.",
        "Key metrics to track: engagement rate, reach, impressions, click-through rate, conversions, and follower growth.",
        "Regularly review analytics to understand what content performs best and adjust your strategy accordingly."
    ]},

    {"intent": "influencer_marketing", "patterns": [
        "influencer marketing",
        "how to work with influencers",
        "social media influencers"
    ], "response": [
        "Choose influencers aligned with your brand values and target audience. Look for engagement rates over follower count.",
        "Micro-influencers often have higher engagement and lower costs compared to celebrity influencers. Focus on authenticity and relevance.",
        "Develop clear collaboration guidelines, provide creative freedom, and track performance of influencer partnerships."
    ]},

    {"intent": "social_media_tools", "patterns": [
        "social media management tools",
        "best tools for social media marketing",
        "tools to help with social media"
    ], "response": [
        "Recommended tools include Hootsuite, Buffer, Sprout Social for scheduling, Canva for design, Later for Instagram planning, and Google Analytics for tracking.",
        "Use scheduling tools to maintain consistent posting, design tools to create engaging visuals, and analytics tools to measure performance.",
        "Automation can help, but always maintain a personal touch in your social media interactions."
    ]}
]

# Prepare training data: Extract patterns and responses
patterns = []
responses = []
intents = []

for intent in training_data:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        responses.append(random.choice(intent['response']))
        intents.append(intent['intent'])

# Vectorize the input using TF-IDF
vectorizer = TfidfVectorizer()

# Create a machine learning pipeline: TF-IDF Vectorizer + Naive Bayes Classifier
classifier = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
classifier.fit(patterns, intents)


# Function to get a response from the trained model
def get_response(user_input):
    # Predict the intent
    intent = classifier.predict([user_input])[0]

    # Find the corresponding response from the training data
    for intent_data in training_data:
        if intent_data['intent'] == intent:
            return random.choice(intent_data['response'])

    return "Sorry, I don't understand that."


# Main chatbot loop
def chatbot():
    print("Chatbot: Hello! How can I assist you today? (Type 'bye' to exit.)")

    while True:
        user_input = input("You: ")

        # If the user says 'bye', exit the loop
        if 'bye' in user_input.lower():
            print("Chatbot: Goodbye!")
            break

        # Get and print the chatbot's response
        response = get_response(user_input)
        print("Chatbot:", response)


# Start the chatbot
if __name__ == "__main__":
    chatbot()
