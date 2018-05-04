from fileProcess import FileStore,FileReader,DataLoader
import settings
from pyvi import ViTokenizer
from gensim import corpora, matutils
import os 
from random import randint

class NLP(object):
    def __init__(self, text = None):
        self.text = text
        self.__set_stopwords()

    def __set_stopwords(self):
        self.stopwords = FileReader(settings.STOP_WORDS).read_stopwords()

    def segmentation(self):
        return ViTokenizer.tokenize(self.text)

    def split_words(self):
        text = self.segmentation()
        try:
            return [x.strip(settings.SPECIAL_CHARACTER).lower() for x in text.split()]
        except TypeError:
            return []

    def get_words_feature(self):
        split_words = self.split_words()
        return [word for word in split_words if word.encode('utf-8') not in self.stopwords]

class FeatureExtraction(object):
    def __init__(self, data):
        self.data = data

    def __build_dictionary(self):
        print 'Building dictionary'
        dict_words = []
        i = 0
        for text in self.data:
            i += 1
            print "Step {} / {}".format(i, len(self.data))
            words = NLP(text = text['content']).get_words_feature()
            dict_words.append(words)
        FileStore(filePath=settings.DICTIONARY_PATH).store_dictionary(dict_words)

    def __load_dictionary(self):
        if os.path.exists(settings.DICTIONARY_PATH) == False:
            self.__build_dictionary()
        self.dictionary = FileReader(settings.DICTIONARY_PATH).load_dictionary()

    def __build_dataset(self):
        print 'Building dataset'
        self.features = []
        self.labels = []
        i = 0
        for d in self.data:
            i += 1
            print "Step {} / {}".format(i, len(self.data))
            self.features.append(self.get_dense(d['content']))
            self.labels.append(d['category'])

    def get_dense(self, text):
        self.__load_dictionary()
        words = NLP(text).get_words_feature()
        # Bag of words
        vec = self.dictionary.doc2bow(words)
        dense = list(matutils.corpus2dense([vec], num_terms=len(self.dictionary)).T[0])
        return dense

    def get_data_and_label(self):
        self.__build_dataset()
        return self.features, self.labels

    def read_feature(self):
        return self.data['features'] , self.data['labels']

if __name__ == '__main__':
    json_train = DataLoader(dataPath=settings.DATA_TRAIN_PATH).get_json()
    FileStore(filePath=settings.DATA_TRAIN_JSON, data=json_train).store_json()
    json_test = DataLoader(dataPath=settings.DATA_TEST_PATH).get_json()
    FileStore(filePath=settings.DATA_TEST_JSON, data=json_test).store_json()
    print 'PreProcess Data Done!'