from flask import Flask,render_template,request, session, redirect, url_for
from flask_pymongo import PyMongo
import pickle
import numpy as np
import bcrypt
import pandas as pd
import pymongo

import pandas as  pd
import spacy
    
import seaborn as sns
import string

from tqdm import tqdm
from textblob import TextBlob
    
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import re
   
    
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
    
    
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction import DictVectorizer
    
import swifter






popular_df = pickle.load(open('popular.pkl','rb'))
pt = pickle.load(open('pt.pkl','rb'))
books = pickle.load(open('books.pkl','rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl','rb'))
df = pd.read_json("books.json")
df.drop(df[(df['categories'] == 'Fiction')].index, inplace=True)

sem6_df = pickle.load(open('new.pkl','rb'))
sem6_df_rom = pickle.load(open('rom.pkl','rb'))
sem6_df_ms_tr = pickle.load(open('mystery_triller.pkl','rb'))
sem6_lit_fic = pickle.load(open('literature_fiction.pkl','rb'))
sem6_eng =pickle.load(open('engineering.pkl','rb'))
sem6_manga = pickle.load(open('manga.pkl','rb'))

app = Flask(__name__)

app.secret_key = "testing"
client = pymongo.MongoClient("mongodb+srv://ruchirrao04:ruchir03@cluster0.coezmw7.mongodb.net/?retryWrites=true&w=majority")

db = client.get_database('total_records')

records = db.register

@app.route('/')
def index():
    return render_template('index.html',
                           book_name = list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values)
                           )


@app.route('/new')
def neww():
    return render_template('new.html',
                           book_name = list(sem6_df['title'].values),
                           author=list(sem6_df['author'].values),
                           image=list(sem6_df['image_link'].values),
                           publisher=list(sem6_df['publisher'].values),
                           language=list(sem6_df['language'].values),
                           rating=list(sem6_df['rating'].values)
                           )

@app.route('/romance')
def romm():
    return render_template('romance.html',
                           book_name = list(sem6_df_rom['title'].values),
                           author=list(sem6_df_rom['author'].values),
                           image=list(sem6_df_rom['image_link'].values),
                           publisher=list(sem6_df_rom['publisher'].values),
                           language=list(sem6_df_rom['language'].values),
                           rating=list(sem6_df_rom['rating'].values)
                           )


@app.route('/mystery_triller')
def mystrs():
    return render_template('mys_tr.html',
                           book_name = list(sem6_df_ms_tr['title'].values),
                           author=list(sem6_df_ms_tr['author'].values),
                           image=list(sem6_df_ms_tr['image_link'].values),
                           publisher=list(sem6_df_ms_tr['publisher'].values),
                           language=list(sem6_df_ms_tr['language'].values),
                           rating=list(sem6_df_ms_tr['rating'].values)
                           )

@app.route('/literature_fiction')
def litfic():
    return render_template('lit_fic.html',
                           book_name = list(sem6_lit_fic['title'].values),
                           author=list(sem6_lit_fic['author'].values),
                           image=list(sem6_lit_fic['image_link'].values),
                           publisher=list(sem6_lit_fic['publisher'].values),
                           language=list(sem6_lit_fic['language'].values),
                           rating=list(sem6_lit_fic['rating'].values)
                           )

@app.route('/engineering')
def engg():
    return render_template('engine.html',
                           book_name = list(sem6_eng['title'].values),
                           author=list(sem6_eng['author'].values),
                           image=list(sem6_eng['image_link'].values),
                           publisher=list(sem6_eng['publisher'].values),
                           language=list(sem6_eng['language'].values),
                           rating=list(sem6_eng['rating'].values)
                           )


@app.route('/manga')
def mangg():
    return render_template('manga.html',
                           book_name = list(sem6_manga['title'].values),
                           author=list(sem6_manga['author'].values),
                           image=list(sem6_manga['image_link'].values),
                           publisher=list(sem6_manga['publisher'].values),
                           language=list(sem6_manga['language'].values),
                           rating=list(sem6_manga['rating'].values)
                           )

@app.route("/register", methods=['post', 'get'])
def register():
    message = ''
    
    if "email" in session:
        return redirect(url_for("logged_in"))
    if request.method == "POST":
        user = request.form.get("fullname")
        email = request.form.get("email")
        password1 = request.form.get("password1")
        password2 = request.form.get("password2")
        
        user_found = records.find_one({"name": user})
        email_found = records.find_one({"email": email})
        if user_found:
            message = 'There already is a user by that name'
            return render_template('register.html', message=message)
        if email_found:
            message = 'This email already exists in database'
            return render_template('register.html', message=message)
        if password1 != password2:
            message = 'Passwords should match!'
            return render_template('register.html', message=message)
        else:
            #hash the password and encode it
            hashed = bcrypt.hashpw(password2.encode('utf-8'), bcrypt.gensalt())
            #assing them in a dictionary in key value pairs
            user_input = {'name': user, 'email': email, 'password': hashed}
            #insert it in the record collection
            records.insert_one(user_input)
            
            #find the new created account and its email
            user_data = records.find_one({"email": email})
            new_email = user_data["email"]
            #if registered redirect to logged in as the registered user
            return redirect(url_for("logged_in"))
    return render_template('register.html')

@app.route("/login", methods=["POST", "GET"])
def login():
    message = 'Please login to your account'
    if "email" in session:
        return redirect(url_for("logged_in"))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        #check if email exists in database
        email_found = records.find_one({"email": email})
        if email_found:
            email_val = email_found['email']
            passwordcheck = email_found['password']
            #encode the password and check if it matches
            if bcrypt.checkpw(password.encode('utf-8'), passwordcheck):
                session["email"] = email_val
                return redirect(url_for('logged_in'))
            else:
                if "email" in session:
                    return redirect(url_for("logged_in"))
                message = 'Wrong password'
                return render_template('login.html', message=message)
        else:
            message = 'Email not found'
            return render_template('login.html', message=message)
    return render_template('login.html', message=message)

@app.route('/logged_in',methods=["POST", "GET"])
def logged_in():
    if "email" in session:
        email = session["email"]
        return render_template('logged_in.html', email=email,
                           book_name = list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values))
   
    else:
        return redirect(url_for("login"))

@app.route("/logout", methods=["POST", "GET"])
def logout():
    if "email" in session:
        session.pop("email", None)
        return render_template("signout.html")
    else:
        return render_template('signout.html')

@app.route('/category')
def categ():
    return render_template('category.html')


@app.route('/about_us')
def sigmas():
    return render_template('about_us.html')

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books',methods=['post'])
def recommend():
    user_input = request.form.get('user_input')

    from difflib import SequenceMatcher

    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()
    
    
    keys = []
    values = []
    dicts ={}
    for i in pt.index:
            keys.append(i)
            values.append(similar(user_input,i))
    for i in range(len(keys)):
        dicts[keys[i]] = values[i]
            
    z = max(dicts.values())

    for i in pt.index:
        if similar(user_input,i) == z:
                ans = str(i)
    
    # for i in pt.index:
    #     if similar(user_input,i) >= 0.7:
    #         ans = str(i)

    index = np.where(pt.index == ans)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]
    r_books =  pd.read_csv('books.csv', index_col = "Book-Title")
    auth_books =  pd.read_csv('books.csv', index_col = "Book-Author")

    
    og_book = r_books.loc[ans]
    x = og_book['Book-Author']
    y =  og_book['Image-URL-M']
    if type(x) is str:
        og_list = [ans, x, y]
    else:
        og_list = [ans, x[1], y[1]]
    
   
    og_auth = auth_books.loc[x]
    z = og_auth['Book-Title']
    v =  og_auth['Image-URL-M']
    # auth_year = og_auth['Year-Of-Publication']
    # print(auth_year.head())
 

    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

        data.append(item)
    
    auth_data = []
    for i in range(4,16):
        auth_list = []
        auth_list.append(z[i])
        if type(x) is str:
            auth_list.append(x)
        else:
            auth_list.append(x[1])
        auth_list.append(v[i])
        auth_data.append(auth_list)    

    print(data)

    return render_template('recommend.html',data=data, og_list = og_list, auth_data = auth_data)


@app.route('/recommend_by_content')
def recommend_nlp():
    return render_template('recommend_nlp.html')


@app.route('/recommend_books_con',methods=['post'])
def recommend_cont():
    user_input = request.form.get('user_input')
    
    stop_words_ = set(stopwords.words('english'))
    wn = WordNetLemmatizer()
    my_sw = ['make', 'amp',  'news','new' ,'time', 'u','s', 'photos',  'get', 'say']

    def black_txt(token):
        return  token not in stop_words_ and token not in list(string.punctuation)  and len(token)>2 and token not in my_sw
    
    def clean_txt(text):
        clean_text = []
        clean_text2 = []
        text = re.sub("'", "",text)
        text=re.sub("(\\d|\\W)+"," ",text)    
        clean_text = [ wn.lemmatize(word, pos="v") for word in word_tokenize(text.lower()) if black_txt(word)]
        clean_text2 = [word for word in clean_text if black_txt(word)]
        return " ".join(clean_text2)
    def subj_txt(text):
        return  TextBlob(text).sentiment[1]

    def polarity_txt(text):
        return TextBlob(text).sentiment[0]

    def len_text(text):
        if len(text.split())>0:
            return len(set(clean_txt(text).split()))/ len(text.split())
        else:
            return 0
    df['text'] = df['title']  +  " " + df['description']

    df['text'] = df['text'].swifter.apply(clean_txt)
    df['polarity'] = df['text'].swifter.apply(polarity_txt)
    df['subjectivity'] = df['text'].swifter.apply(subj_txt)
    df['len'] = df['text'].swifter.apply(lambda x: len(x))

    X = df[['text', 'polarity', 'subjectivity','len']]
    y =df['categories']

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(X, y)
    v = dict(zip(list(y), df['categories'].to_list()))

    text_clf = Pipeline([
     ('vect', CountVectorizer(analyzer="word", stop_words="english")),
     ('tfidf', TfidfTransformer(use_idf=True)),
     ('clf', MultinomialNB(alpha=.01)),
     ])
    
    text_clf.fit(x_train['text'].to_list(), list(y_train))

    import pickle
    with open('model.pkl','wb') as f:
        pickle.dump(text_clf,f)

    with open('model.pkl', 'rb') as f:
        clf2 = pickle.load(f)
    docs_new = [user_input]
    predicted = clf2.predict(docs_new)
    cat = v[predicted[0]]
    cat_books =  pd.read_csv('nlp.csv', index_col = "categories")
    tit_books =  pd.read_csv('nlp.csv', index_col = "title")
    og_book = cat_books.loc[cat]
    tit = og_book['title']
    tumb =  og_book['thumbnail']
    autho = og_book['authors']
    data = []

    
    from difflib import SequenceMatcher
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()
    
    keys = []
    values = []
    dicts ={}
    for i in tit:
            keys.append(i)
            values.append(similar(user_input,i))
    for i in range(len(keys)):
        dicts[keys[i]] = values[i]
            
    z = max(dicts.values())

    for i in tit:
        if similar(user_input,i) == z:
                print(i) 
                org_book = tit_books.loc[i]
                x = org_book['authors']
                y = org_book['thumbnail']
                listt = [i,x,y]


        
    for i in range(28):
        item = []
        item.append(tit[i])
        item.append(autho[i])
        item.append(tumb[i])
        data.append(item)
 

    
    return render_template('recommend_nlp.html',data = data,listt = listt )


@app.route('/author')
def recommend_auth():
    return render_template('authorr.html')


@app.route('/recommend_auth',methods=['post'])
def recomm_auth():
    user_input = request.form.get('user_input')

    from difflib import SequenceMatcher

    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()
    auth_books =  pd.read_csv('nlp.csv', index_col = "authors")
    auth = df['authors']
        
    keys = []
    values = []
    dicts ={}
    for i in auth:
            keys.append(i)
            values.append(similar(user_input,i))
    for i in range(len(keys)):
        dicts[keys[i]] = values[i]
            
    z = max(dicts.values())
    auth_data = []
    for i in auth:
        if similar(user_input,i) == z:
                authhh = i
                og_auth = auth_books.loc[i]
                a_tit = og_auth['title']
                a_thum =  og_auth['thumbnail']
                if type(a_tit) is str:
                    auth_data = [[a_tit,authhh,a_thum]]
                elif len(a_tit) == 1:
                    auth_data = [[a_tit[0],authhh,a_thum[0]]]       
                else:
                    for i in range(len(a_tit)):
                        auth_list = []
                        auth_list.append(a_tit[i])
                        auth_list.append(authhh)
                        auth_list.append(a_thum[i])
                        auth_data.append(auth_list)

    return render_template('authorr.html', auth_data = auth_data ,a_tit = a_tit,a_thum=a_thum, authhh = authhh)





if __name__ == '__main__':
    app.run(debug=True)