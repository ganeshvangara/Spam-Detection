import time
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
    # print(inp)
    inp=pd.Series(inp)
    # print("series",inp)
    inp =inp.apply(stemmer)
    # print("stemmer",inp)
    inp_prep=inp.apply(text_preprocess)
    # print("prepppp",inp_prep)

    inp_test=vect.transform(inp_prep)
    # print("inp_test",inp_test)
    inp_pred=model.predict(inp_test)
    print("Result:::")

    if inp_pred==1:
        print("spam")
        return "Spam"
    else:
        print("Not Spam")

        return "Not Spam"


# model.predict(vectorizer.transform(inp)[0])

inp=["Please call our customer service representative on 0800 169 6031 between 10am-9pm as you have WON a guaranteed å£1000 cash or å£5000 prize"]
# is_spam(inp)



def start(updater, context): 
	updater.message.reply_text("Welcome to the Spam classification bot!")

def help_(updater, context): 
	updater.message.reply_text("Send the message you want check for spam \n From now all the messages will be Classified automatically")

# def rem_user(updater, context):

def message(updater, context):
    msg = updater.message.text
    user_id =updater.message.from_user.id
    chat_id=updater.message.chat_id
    # admin=updater.message.get_chat_administrators(chat_id)
    name  = updater.message.from_user.first_name
    print("message rcvd:\n",msg, "\nfrom:\nuser id::", user_id, "\tname ::" ,name)

    if(is_spam(msg)=="Spam" ) :
         updater.message.reply_text("Spam Message detected")
        #  print(name," is removed because of spam ")
        #  m = context.bot.kick_chat_member(
        # chat_id, 
        # user_id=user_id
        # )
        #  rm_msg=str(name)+"  removed because of spam "
        #  context.bot.send_message(chat_id=updater.message.chat_id, text=rm_msg)
 


        #  updater.sendMessage("User kicked out" ,chat_id)
        #  updater.message.reply_text(" is removed because of spam ")
         
    
	# message.__name__
    else:
        updater.message.reply_text("not spam" )
        # context.bot.send_message(chat_id=updater.message.chat_id, text=" not spammmm  ")
        # updater.send_message("User not     kicked out" ,chat_id)


    # updater.message.reply_text(" is removed because of spam ")

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
# dispatcher.add_handler(CommandHandler("Do you want to remove", rem_user))

dispatcher.add_handler(MessageHandler(Filters.text, message))

# dispatcher.add_handler(MessageHandler(Filters.photo, image))
updater.start_polling()
updater.idle()
