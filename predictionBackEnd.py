from flask import Flask, redirect, render_template, request
from wtforms import Form, TextAreaField, SelectField, validators
from nltk.corpus import stopwords
import numpy as np
import pickle
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
class inputform(Form):
    traitselection=SelectField('', [validators.DataRequired()], choices=[
        ('OCEAN','the Big Five'),
        ('MBTI','MBTI')
    ])
    modelselection=SelectField('', [validators.DataRequired()], choices=[
        ('LR','Logistic Regression'),
        ('DT','Decision Tree'),
        ('RF','Random Forest'),
        ('KNN','K nearest neighbor'),
        ('MLP','MLP')
    ])
    textinput=TextAreaField('Input text here',[validators.DataRequired(), validators.length(min=15)])

stopwords=['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 
           'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn',
            "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further',
            'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here',
            'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll","i'm", 'in', 'into', 'is', 'isn', "isn't", 
            'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn',
            "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 
            'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', 
            "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 
            'these','they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was',
              'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 
              'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 
              'yourselves', "you've"]
def stopword(text):
    text_ = []
    for i in text.split():
        if i not in stopwords:
            text_.append(i)
    return ' '.join(text_)
def preprocessor(text):
    """ Cleans raw text by removing HTML and punctuation, but preserving emoticons. """
    # Find any HTML tags (e.g., <br />, </a>) and replace them with an empty string
    text = re.sub('<[^>]*>', '', text)
    # Find all emoticons in the text (e.g., :), :(, :D, :-P) and store them in a list
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # 1. Lowercase the entire text.
    # 2. Find any non-word characters (punctuation) using '[\W]+' and replace them with a space.
    # 3. Reattach the saved emoticons to the end of the string, removing their 'nose' character ('-').
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text
def classify(trait,model,text):
    text=preprocessor(text)
    text=stopword(text)
    if trait=='MBTI':
        tfidf = pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'tfidf.pkl'), 'rb'))
        text=tfidf.transform([text])
        if model=='LR':
            clf_IE= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'MBTIclassifier_LR_IE.pkl'), 'rb'))
            clf_NS= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'MBTIclassifier_LR_NS.pkl'), 'rb'))
            clf_TF= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'MBTIclassifier_LR_TF.pkl'), 'rb'))
            clf_JP= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'MBTIclassifier_LR_JP.pkl'), 'rb'))
            pred_IE=clf_IE.predict(text)[0]
            pred_NS=clf_NS.predict(text)[0]
            pred_TF=clf_TF.predict(text)[0]
            pred_JP=clf_JP.predict(text)[0]
            proba_IE=np.max(clf_IE.predict_proba(text))
            proba_NS=np.max(clf_NS.predict_proba(text))
            proba_TF=np.max(clf_TF.predict_proba(text))
            proba_JP=np.max(clf_JP.predict_proba(text))
            return pred_IE, pred_NS, pred_TF, pred_JP, proba_IE, proba_NS, proba_TF, proba_JP
        elif model=='DT':
            clf_IE= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'MBTIclassifier_DT_IE.pkl'), 'rb'))
            clf_NS= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'MBTIclassifier_DT_NS.pkl'), 'rb'))
            clf_TF= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'MBTIclassifier_DT_TF.pkl'), 'rb'))
            clf_JP= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'MBTIclassifier_DT_JP.pkl'), 'rb'))
            pred_IE=clf_IE.predict(text)[0]
            pred_NS=clf_NS.predict(text)[0]
            pred_TF=clf_TF.predict(text)[0]
            pred_JP=clf_JP.predict(text)[0]
            proba_IE=np.max(clf_IE.predict_proba(text))
            proba_NS=np.max(clf_NS.predict_proba(text))
            proba_TF=np.max(clf_TF.predict_proba(text))
            proba_JP=np.max(clf_JP.predict_proba(text))
            return pred_IE, pred_NS, pred_TF, pred_JP, proba_IE, proba_NS, proba_TF, proba_JP
        elif model=='RF':
            clf_IE= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'MBTIclassifier_RF_IE.pkl'), 'rb'))
            clf_NS= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'MBTIclassifier_RF_NS.pkl'), 'rb'))
            clf_TF= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'MBTIclassifier_RF_TF.pkl'), 'rb'))
            clf_JP= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'MBTIclassifier_RF_JP.pkl'), 'rb'))
            pred_IE=clf_IE.predict(text)[0]
            pred_NS=clf_NS.predict(text)[0]
            pred_TF=clf_TF.predict(text)[0]
            pred_JP=clf_JP.predict(text)[0]
            proba_IE=np.max(clf_IE.predict_proba(text))
            proba_NS=np.max(clf_NS.predict_proba(text))
            proba_TF=np.max(clf_TF.predict_proba(text))
            proba_JP=np.max(clf_JP.predict_proba(text))
            return pred_IE, pred_NS, pred_TF, pred_JP, proba_IE, proba_NS, proba_TF, proba_JP
        elif model=='KNN':
            clf_IE= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'MBTIclassifier_KNN_IE.pkl'), 'rb'))
            clf_NS= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'MBTIclassifier_KNN_NS.pkl'), 'rb'))
            clf_TF= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'MBTIclassifier_KNN_TF.pkl'), 'rb'))
            clf_JP= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'MBTIclassifier_KNN_JP.pkl'), 'rb'))
            pred_IE=clf_IE.predict(text)[0]
            pred_NS=clf_NS.predict(text)[0]
            pred_TF=clf_TF.predict(text)[0]
            pred_JP=clf_JP.predict(text)[0]
            proba_IE=np.max(clf_IE.predict_proba(text))
            proba_NS=np.max(clf_NS.predict_proba(text))
            proba_TF=np.max(clf_TF.predict_proba(text))
            proba_JP=np.max(clf_JP.predict_proba(text))
            return pred_IE, pred_NS, pred_TF, pred_JP, proba_IE, proba_NS, proba_TF, proba_JP
        elif model=='MLP':
            clf_IE= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'MBTIclassifier_MLP_IE.pkl'), 'rb'))
            clf_NS= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'MBTIclassifier_MLP_NS.pkl'), 'rb'))
            clf_TF= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'MBTIclassifier_MLP_TF.pkl'), 'rb'))
            clf_JP= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'MBTIclassifier_MLP_JP.pkl'), 'rb'))
            pred_IE=clf_IE.predict(text)[0]
            pred_NS=clf_NS.predict(text)[0]
            pred_TF=clf_TF.predict(text)[0]
            pred_JP=clf_JP.predict(text)[0]
            proba_IE=np.max(clf_IE.predict_proba(text))
            proba_NS=np.max(clf_NS.predict_proba(text))
            proba_TF=np.max(clf_TF.predict_proba(text))
            proba_JP=np.max(clf_JP.predict_proba(text))
            return pred_IE, pred_NS, pred_TF, pred_JP, proba_IE, proba_NS, proba_TF, proba_JP
    elif trait=='OCEAN':
        tfidf = pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'tfidf_OCEAN.pkl'), 'rb'))
        text=tfidf.transform([text])
        if model=='LR':
            clf_A= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'OCEANclassifier_LR_A.pkl'), 'rb'))
            clf_C= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'OCEANclassifier_LR_C.pkl'), 'rb'))
            clf_E= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'OCEANclassifier_LR_E.pkl'), 'rb'))
            clf_N= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'OCEANclassifier_LR_N.pkl'), 'rb'))
            clf_O= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'OCEANclassifier_LR_O.pkl'), 'rb'))
            pred_A=clf_A.predict(text)[0]
            pred_C=clf_C.predict(text)[0]
            pred_E=clf_E.predict(text)[0]
            pred_N=clf_N.predict(text)[0]
            pred_O=clf_O.predict(text)[0]
            proba_A=np.max(clf_A.predict_proba(text))
            proba_C=np.max(clf_C.predict_proba(text))
            proba_E=np.max(clf_E.predict_proba(text))
            proba_N=np.max(clf_N.predict_proba(text))
            proba_O=np.max(clf_O.predict_proba(text))
            return pred_A, pred_C, pred_E, pred_N, pred_O, proba_A, proba_C, proba_E, proba_N, proba_O
        elif model=='DT':
            clf_A= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'OCEANclassifier_DT_A.pkl'), 'rb'))
            clf_C= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'OCEANclassifier_DT_C.pkl'), 'rb'))
            clf_E= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'OCEANclassifier_DT_E.pkl'), 'rb'))
            clf_N= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'OCEANclassifier_DT_N.pkl'), 'rb'))
            clf_O= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'OCEANclassifier_DT_O.pkl'), 'rb'))
            pred_A=clf_A.predict(text)[0]
            pred_C=clf_C.predict(text)[0]
            pred_E=clf_E.predict(text)[0]
            pred_N=clf_N.predict(text)[0]
            pred_O=clf_O.predict(text)[0]
            proba_A=np.max(clf_A.predict_proba(text))
            proba_C=np.max(clf_C.predict_proba(text))
            proba_E=np.max(clf_E.predict_proba(text))
            proba_N=np.max(clf_N.predict_proba(text))
            proba_O=np.max(clf_O.predict_proba(text))
        elif model=='KNN':
            clf_A= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'OCEANclassifier_KNN_A.pkl'), 'rb'))
            clf_C= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'OCEANclassifier_KNN_C.pkl'), 'rb'))
            clf_E= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'OCEANclassifier_KNN_E.pkl'), 'rb'))
            clf_N= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'OCEANclassifier_KNN_N.pkl'), 'rb'))
            clf_O= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'OCEANclassifier_KNN_O.pkl'), 'rb'))
            pred_A=clf_A.predict(text)[0]
            pred_C=clf_C.predict(text)[0]
            pred_E=clf_E.predict(text)[0]
            pred_N=clf_N.predict(text)[0]
            pred_O=clf_O.predict(text)[0]
            proba_A=np.max(clf_A.predict_proba(text))
            proba_C=np.max(clf_C.predict_proba(text))
            proba_E=np.max(clf_E.predict_proba(text))
            proba_N=np.max(clf_N.predict_proba(text))
            proba_O=np.max(clf_O.predict_proba(text))
            return pred_A, pred_C, pred_E, pred_N, pred_O, proba_A, proba_C, proba_E, proba_N, proba_O
        elif model=='MLP':
            clf_A= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'OCEANclassifier_MLP_A.pkl'), 'rb'))
            clf_C= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'OCEANclassifier_MLP_C.pkl'), 'rb'))
            clf_E= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'OCEANclassifier_MLP_E.pkl'), 'rb'))
            clf_N= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'OCEANclassifier_MLP_N.pkl'), 'rb'))
            clf_O= pickle.load(open(os.path.join("PersonalityClassifier",'pkl_objects', 'OCEANclassifier_MLP_O.pkl'), 'rb'))
            pred_A=clf_A.predict(text)[0]
            pred_C=clf_C.predict(text)[0]
            pred_E=clf_E.predict(text)[0]
            pred_N=clf_N.predict(text)[0]
            pred_O=clf_O.predict(text)[0]
            proba_A=np.max(clf_A.predict_proba(text))
            proba_C=np.max(clf_C.predict_proba(text))
            proba_E=np.max(clf_E.predict_proba(text))
            proba_N=np.max(clf_N.predict_proba(text))
            proba_O=np.max(clf_O.predict_proba(text))
            return pred_A, pred_C, pred_E, pred_N, pred_O, proba_A, proba_C, proba_E, proba_N, proba_O
app=Flask(__name__)
@app.route('/')
def index():
    form=inputform(request.form)
    return render_template('predictionFrontPage.html',form=form)
@app.route('/results', methods=['POST'])
def results():
    form=inputform(request.form)
    if request.method=='POST' and form.validate():
        trait=request.form['traitselection']
        model=request.form['modelselection']
        text=request.form['textinput']
        if trait=='MBTI':
            pred_IE, pred_NS, pred_TF, pred_JP, proba_IE, proba_NS, proba_TF, proba_JP=classify(trait,model,text)
            traits={'I':'Introverted','E':'Extraverted','N':'Intuitive','S':'Observant','T':'Thinking','F':'Feeling','J':'Judging','P':'Prospecting'}
            return render_template('predictionResultPage_MBTI.html', pred_IE=pred_IE, pred_NS=pred_NS, pred_TF=pred_TF, pred_JP=pred_JP,
                                proba_IE=int(proba_IE*100), proba_NS=int(proba_NS*100), proba_TF=int(proba_TF*100), proba_JP=int(proba_JP*100), traits=traits)
        elif trait=='OCEAN':
            pred_A, pred_C, pred_E, pred_N, pred_O, proba_A, proba_C, proba_E, proba_N, proba_O=classify(trait,model,text)
            traits={1:'more',0:'less'}
            return render_template('predictionResultPage_OCEAN.html', pred_A=pred_A, pred_C=pred_C, pred_E=pred_E, pred_N=pred_N, pred_O=pred_O,
                                   proba_A=int(proba_A*100), proba_C=int(proba_C*100), proba_E=int(proba_E*100), proba_N=int(proba_N*100), proba_O=int(proba_O*100), traits=traits)
    return render_template('predictionFrontPage.html',form=form)
if __name__=='__main__':
    app.run()