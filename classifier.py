# Written by Edmund Hee - 20160618

import os, json, re, sys, numpy
from nltk.stem import WordNetLemmatizer
from optparse import OptionParser
from stopword import Stopword
import cPickle as pickle
from lda import LDA

class Classifier:

    def __init__(self, options):
        self.options = options
        self.file_dir = "./build/"
        self.labels= []
        self.corpus = []

        if not os.path.exists(self.file_dir):
            os.makedirs("build")

        self.stopwords = self.get_stopwords()


    def train_model(self, filename, model_name):
        self.create_label_corpus(filename)
        self.lda = LDA(self.options.K, self.options.alpha, self.options.beta)
        self.lda.set_corpus(self.labelset, self.corpus, self.labels)
        print "M=%d, V=%d, L=%d, K=%d" % (len(self.corpus), len(self.lda.vocas), len(self.labelset), self.options.K)
        for index in range(self.options.iteration):
            sys.stderr.write("-- %d : %.4f\n" % (index, self.lda.perplexity()))
        print "perplexity : %.4f" % self.lda.perplexity()
        phi = self.lda.phi()
        theta = self.lda.theta()
        new_stopword = []
        for k, label in enumerate(self.labelset):
            print "\n-- label %d : %s" % (k, label)
            for w in numpy.argsort(-phi[k]):
                print "%s: %f" % (self.lda.vocas[w], phi[k,w])
        self.save_model(model_name)

    def lemmatize(self, string):
        return WordNetLemmatizer().lemmatize(string, pos='v')

    def create_label_corpus(self,filename):
        with open(os.path.join(self.file_dir,filename)) as model:
            for row in model:
                label_class_list = []
                selected_words = []

                split_row = row.lower().split("\"|\"")
                label_array = self.filter_split(split_row[0])
                # Create Unicoded label_type
                for label_type in self.filter_split(split_row[1]):
                    label_class_list.append(unicode(label_type,"utf-8"))

                for word in label_array:
                    lemmatized_word = self.lemmatize(word)
                    if word not in self.stopwords and len(word) > 2 and not bool(re.search(r'\d',lemmatized_word)) and lemmatized_word not in self.stopwords:
                        selected_words.append(lemmatized_word)

                self.corpus.append(selected_words)
                self.labels.append(label_class_list)
                self.labelset = list(set(reduce(list.__add__, self.labels)))

    def filter_split(self,label):
        return re.sub(r'\W+',' ',label).split()

    def classify(self,model_name,label):
        self.lda = self.load_model(model_name)
        self.stopwords = self.get_stopwords()
        result_vector = numpy.zeros(self.lda.K)
        phi = self.lda.phi()
        label_array = self.filter_split(label)

        for word in label_array:
            for r in range(self.lda.K):
                lemmatized_word = self.lemmatize(word)
                if word not in self.stopwords and len(word) > 2 and not bool(re.search(r'\d',lemmatized_word)) and lemmatized_word not in self.stopwords and lemmatized_word in self.lda.vocas_id:
                    result_vector[r] += phi[r,self.lda.vocas_id[lemmatized_word]]

        result = 0
        if result_vector.argmax() == 0:
            v = max(n for n in result_vector if n != max(result_vector))
            result = numpy.argwhere(result_vector == v)
        else:
            result = result_vector.argmax()
        print self.lda.labelmap.keys()[self.lda.labelmap.values().index(result)]
        return self.lda.labelmap.keys()[self.lda.labelmap.values().index(result)]

    def save_model(self, model_name):
        with open(os.path.join(self.file_dir,model_name + "_trained.p"),'wb') as model_file:
            pickle.dump(self.lda,model_file,protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self,model_name):
        if os.path.isfile(os.path.join(self.file_dir,model_name+ "_trained.p")):
            with open(os.path.join(self.file_dir,model_name + "_trained.p"),'rb') as model_file:
                return pickle.load(model_file)
        else:
            print "Trained model for %s is not found in \"%s\" directory" % ((model_name), (file_dir))
            print "Please train the model"

    def get_stopwords(self):
        return Stopword(self.file_dir).get_stopwords()


parser = OptionParser()
parser.add_option("-f", dest="filename", type="string", help="File Name")
parser.add_option("-m", dest="model_name", type="string", help="Model name")
parser.add_option("-l", dest="label", type="string", help="Label to pass for classification")
parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.005)
parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.005)
parser.add_option("-k", dest="K", type="int", help="number of topics", default=10)
parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
parser.add_option("-t", dest="train", help="train data", default= False, action='store_true')
parser.add_option("-c", dest="classify", help="Classify data", default= False, action='store_true')
parser.add_option("--sos", dest="help", help="Help ME!", default= False, action='store_true')
(options, args) = parser.parse_args()
classifier = Classifier(options)
if options.train:
    classifier.train_model(options.filename,options.model_name)
elif options.classify:
    classifier.classify(options.model_name,options.label)
elif options.help:
    print "--alpha : Set alpha value (default 0.005)"
    print "--beta  : Set beta value (default: 0.005)"
    print "-t : Train with training data in build directory"
    print "-c : Get classification result from trained model"
    print "-m  : Model name of data in build directory"
    print "-l : label to predict (e.g -l \"<TITLE>\")"
    print "-k      : Number of topics (default: 10) * will change according to data set"
    print "-i      : Number of interation (default: 100)"
    print "-n      : Number of sample size (default: 100) * will change according to data set"
    print "-f      : Filename  (e.g -f <FILENAME>) default:headlones_classified.txt"
    print "\n"
    print "To train data"
    print "python classifier.py -m <MODEL_NAME> -f <TEXT_FILE_NAME>.txt -t"
    print "To train classify"
    print "python classifier.py -c -m <MODEL_NAME> -l \"<LABEL TO PASS INTO LDA>\""
else:
    print "Pleaes Select Type: -t for train model -c for classification"
