from fastapi import FastAPI
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pickle

stop_words = stopwords.words("english")
wn = nltk.WordNetLemmatizer()

infile = open("process.p", "rb")
process = pickle.load(infile)
infile.close()

app = FastAPI()

# Nous allons maintenant utiliser Regular Expression pour supprimer certains caractéres. Comme on peut le voir dans notre
# exemple, il reste encore des caractères à supprimer


def clean_text(x):

    # Remove unicode characters
    x = x.encode("ascii", "ignore").decode()
    x = x.replace('\\n', ' ')
    # Remove English contractions
    x = re.sub("\'\w+", '', x)
    # Remove ponctuation but not # (for C# for example)
    x = re.sub('[^\\w\\s#]', '', x)
    # Remove links
    x = re.sub(r'http*\S+', '', x)
    # Remove numbers
    x = re.sub(r'\w*\d+\w*', '', x)
    # Remove extra spaces
    x = re.sub('\s+', ' ', x)
    # Case normalization
    x = x.lower()
    x = x.replace("c #", "c#")
    return x


@app.get("/")
async def Question_Prédictor(
    Question: str
):
    x = BeautifulSoup(Question, "html.parser").get_text()
    x = clean_text(x)
    x = nltk.tokenize.word_tokenize(x)
    x = [word for word in x if word not in stop_words]
    x = [wn.lemmatize(word) for word in x]
    x = process["Tf-idf"].transform([x])
    x = process["RandomForrestClassifier"].predict(x)
    x = process["Multi-Label-Binarizer"].inverse_transform(x)
    return x