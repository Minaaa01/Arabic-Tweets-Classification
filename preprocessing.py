import re
import string
import pyarabic.araby as ab
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# nltk.download('stopwords')
stopwords = stopwords.words('Arabic')
tw = []
tweets = 0


class pre_processing:
    def clean_data(self):
        for tweets in self:
            tashkel_removed = ab.strip_diacritics(tweets)
            emotion_removed = re.sub('['
                                     '(\U0001F600-\U0001F92F|\U0001F300-\U0001F5FF|\U0001F680-\U0001F6FF|\U0001F190-\U0001F1FF|\U00002702'
                                     '-\U000027B0|\U0001F926-\U0001FA9F|\u200d|\u2640-\u2642|\u2600-\u2B55|\u23cf|\u23e9|\u231a|\ufe0f'
                                     ')]+', '', tashkel_removed)
            # remove english words
            eng_removed = re.sub(r'[a-zA-Z]', '', emotion_removed)
            pattern = r'[' + string.punctuation + ']'
            # Remove special characters from the string
            spchar_removed = re.sub(pattern, '', eng_removed)
            # to remove stop words
            text_tokens = word_tokenize(spchar_removed)
            remove_sw = ' '.join([i for i in text_tokens if i not in stopwords])
            # Remove digits
            digit_removed = re.sub("\d+", "", remove_sw)
            tw.append(digit_removed)
        return tw

    def display(self):
        for i in self:
            print(i, sep='\n')

    def count_punct(self):
        count = sum([1 for char in self if char in string.punctuation])
        return round(count / (len(self) - self.count(" ")), 3) * 100
