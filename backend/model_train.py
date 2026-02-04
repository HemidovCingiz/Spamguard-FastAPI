import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

messages = [ #NOT SPAM (50)
            "Hey, how are you today?", "Are we still meeting for lunch?", "Don't forget to bring your notebook", "Can you send me the report?", "I will call you tomorrow", "Let's go to the park this weekend", "Did you finish the homework?", "Please reply to my email", "Are you free tonight?", "Meeting has been rescheduled to 3 PM", "Thank you for your help", "I enjoyed the movie yesterday", "Can we discuss the project later?", "Remember to submit your assignment", "Let's have coffee next week", "Happy birthday! Hope you enjoy your day", "Can you join the meeting online?", "I will send the presentation soon", "See you at the gym later", "Please review the document", "Do you want to go shopping?", "I left the keys on the table", "Are you coming to the party?", "Let's organize the files", "The event starts at 5 PM", "Can you help me with this task?", "I will pick you up at 7", "Please confirm your attendance", "Do you have any questions?", "Let's plan our trip next month", "Your order has been shipped", "Can we meet after work?", "Remember to water the plants", "I will send the invoice today", "See you at the conference", "Can you edit this document?", "Please check the schedule", "I will call you later today", "Let's meet at the coffee shop", "Have you completed the form?", "The weather is great today", "Please send the feedback", "I will send the meeting notes", "Can you review my code?", "Do you want to join us for dinner?", "Let's catch up soon", "I will email the details", "Are you available tomorrow?", "Please send the photos", "I hope you are doing well", 
            # SPAM (50)
            "Win a free iPhone now", "Congratulations! You've won $1000", "Limited time offer, buy now", "Get cheap loans instantly", "Earn money from home", "Click here to claim your prize", "Exclusive deal just for you", "Act now to win big rewards", "Lowest prices guaranteed", "You have been selected for a gift", "Free trial, no credit card required", "Get rich quick with this method", "Don't miss this opportunity", "Special promotion ends today", "Claim your free vacation now", "Urgent! Update your account information", "Increase your followers instantly", "Buy one get one free offer", "Free access to premium content", "This offer expires soon", "Cash bonus waiting for you", "Earn $500 daily working from home", "Click to unlock your reward", "Get your free subscription", "Limited spots available, act fast", "Exclusive access to new products", "Make money while you sleep", "Your account has been compromised", "Free gift card waiting for you", "Click here to win prizes", "Earn cash bonuses today", "Join our free webinar now", "Get a free iPad instantly", "Claim your free coupon", "Special discount just for you", "Win big with this offer", "Free online course available", "Act now to receive your gift", "You are selected for a prize", "Limited edition product available", "Don't miss out on this deal", "Claim your reward instantly", "Earn money online quickly", "Click here for instant access", "Free trial subscription available", "Get your free gift now", "Sign up to claim your bonus", "Limited offer, hurry up", "Earn rewards by clicking here", "Exclusive deal ends today" ]
labels = ["not spam"] * 50 + ["spam"] * 50

clean = [preprocess(m) for m in messages]

vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(clean)

model = ComplementNB(alpha=0.5)
model.fit(X, labels)

joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained & saved.")
