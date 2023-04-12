from telegram.ext import Updater, Filters, CommandHandler, MessageHandler
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer



def stemmer (text):
    text = text.split()
    words = ""
    for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    return words

def text_preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)

vectorizer = TfidfVectorizer()

model = pickle.load(open('svc.pkl','rb'))

vect=pickle.load(open('vector1.pkl','rb'))

def is_spam(inp):
    print(inp)
    inp=pd.Series(inp)
    print("series",inp)
    inp =inp.apply(stemmer)
    print("stemmer",inp)
    inp_prep=inp.apply(text_preprocess)
    print("prepppp",inp_prep)

    inp_test=vect.transform(inp_prep)
    print("inp_test",inp_test)
    inp_pred=model.predict(inp_test)
    print("op",inp_pred)

    if inp_pred==1:
        print("spam")
        return "Spam"
    else:
        print("Not Spam")

        return "Not Spam"


# model.predict(vectorizer.transform(inp)[0])

inp=["Please call our customer service representative on 0800 169 6031 between 10am-9pm as you have WON a guaranteed å£1000 cash or å£5000 prize"]
is_spam(inp)



def start(updater, context): 
	updater.message.reply_text("Welcome to the classification bot!")

def help_(updater, context): 
	updater.message.reply_text("Just send the image you want to classify.")

def message(updater, context):
	msg = updater.message.text
	print(msg)
	
	updater.message.reply_text(is_spam(msg))

# def image(updater, context):
# 	photo = updater.message.photo[-1].get_file()
# 	photo.download("img.jpg"
	# img = cv2.imread("img.jpg")
	# img = cv2.resize(img, (224,224))
	# img = np.reshape(img, (1,224,224,3))
	# pred = np.argmax(model.predict(img))
	# updater.message.reply_text(pred)

updater = Updater("6260509256:AAFUBAi6fBki-WA4-l1yM6S_yxxShFzSNnI",use_context=True)
dispatcher = updater.dispatcher

dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("help", help_))

dispatcher.add_handler(MessageHandler(Filters.text, message))

# dispatcher.add_handler(MessageHandler(Filters.photo, image))
updater.start_polling()
updater.idle()
