import json
from gensim import corpora, matutils
import cPickle
import os
from random import randint

class DataLoader(object):
    def __init__(self, dataPath):
        self.dataPath = dataPath

    def __get_files(self):
        folders = [self.dataPath + folder + '/' for folder in os.listdir(self.dataPath)]
        class_titles = os.listdir(self.dataPath)
        files = {}
        for folder, title in zip(folders, class_titles):
            files[title] = [folder + f for f in os.listdir(folder)]
        self.files = files

    def get_json(self):
        self.__get_files()
        data = []
        for topic in self.files:
            # rand = randint(1, 5)
            rand =1400
            i = 0
            for file in self.files[topic]:
                content = FileReader(filePath=file).content()
                data.append({
                    'category': topic,
                    'content': content
                })
                if i == rand:
                    break
                else:
                    i += 1
        return data

class FileReader(object):
    def __init__(self, filePath, encoder = None):
        self.filePath = filePath
        self.encoder = encoder if encoder != None else 'utf-16le'

    def read(self):
        with open(self.filePath) as f:
            s = f.read()
        return s

    def content(self):
        s = self.read()
        return s.decode(self.encoder)

    def read_json(self):
        with open(self.filePath) as f:
            s = json.load(f)
        return s

    def read_stopwords(self):
        with open(self.filePath, 'r') as f:
            stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
        return stopwords

    def load_dictionary(self):
        return corpora.Dictionary.load_from_text(self.filePath)

class FileStore(object):
    def __init__(self, filePath, data = None):
        self.filePath = filePath
        self.data = data

    def store_json(self):
        with open(self.filePath, 'w') as outfile:
            json.dump(self.data, outfile)

    def store_dictionary(self, dict_words):
        dictionary = corpora.Dictionary(dict_words)
        dictionary.filter_extremes(no_below=100, no_above=0.5)
        dictionary.save_as_text(self.filePath)

    def save_pickle(self,  obj):
        outfile = open(self.filePath, 'wb')
        fastPickler = cPickle.Pickler(outfile, cPickle.HIGHEST_PROTOCOL)
        fastPickler.fast = 1
        fastPickler.dump(obj)
        outfile.close()