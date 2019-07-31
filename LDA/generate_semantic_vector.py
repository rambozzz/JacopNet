import os,sys,re

import numpy


import gensim
from gensim import utils, corpora, models

from LDA.preprocess_text import preprocess
from gensim.test.utils import datapath

NUM_TOPICS = 40
directory = 'LDA'


class SemanticSpaceConverter:
    def __init__(self):
        self.dictionary = self.get_dictionary()
        self.corpus = self.get_corpus()
        self.ldamodel = self.get_model()

    def get_dictionary(self):
        # load id <-> term dictionary
        if not os.path.isfile('../LDA/dictionary.dict'):
            sys.exit('ERR: ID <-> Term dictionary file dictionary.dict not found!')

        print ('Loading id <-> term dictionary from dictionary.dict ...')
        sys.stdout.flush()
        dictionary = corpora.Dictionary.load('../LDA/dictionary.dict')
        print (' Done!')
        # ignore words that appear in less than 20 documents or more than 50% documents
        dictionary.filter_extremes(no_below=20, no_above=0.5)
        return dictionary

    def get_corpus(self):
        # load document-term matrix
        if not os.path.isfile('../LDA/bow.mm'):
            sys.exit('ERR: Document-term matrix file bow.mm not found!')

        print ('Loading document-term matrix from bow.mm ...')
        sys.stdout.flush()
        corpus = gensim.corpora.MmCorpus('../LDA/bow.mm')
        print (' Done!')
        return corpus

    def get_model(self):
        # load LDA model
        if not os.path.isfile('../LDA/ldamodel'+str(NUM_TOPICS)+'.lda'):
            sys.exit('ERR: LDA model file ldamodel'+str(NUM_TOPICS)+'.lda not found!')
        print ('Loading LDA model from file ldamodel'+str(NUM_TOPICS)+'.lda ...')
        sys.stdout.flush()
        ldamodel = models.LdaModel.load('../LDA/ldamodel'+str(NUM_TOPICS)+'.lda')
        print (' Done!')
        return ldamodel

    def generate_constraint(self, raw):
        tokens = preprocess(raw)
        bow_vector = self.dictionary.doc2bow(tokens)
        #lda_vector = ldamodel[bow_vector]
        lda_vector = self.ldamodel.get_document_topics(bow_vector, minimum_probability=None)
        lda_vector = sorted(lda_vector,key=lambda x:x[1],reverse=True)
        topic_prob = {}
        for instance in lda_vector:
            topic_prob[instance[0]] = instance[1]
        constraint = []
        for topic_num in range(0,NUM_TOPICS):
            if topic_num in topic_prob.keys():
              constraint.append(numpy.float64(topic_prob[topic_num]))
              #labels.append(topic_prob[topic_num])
            else:
              constraint.append(0)

        return constraint