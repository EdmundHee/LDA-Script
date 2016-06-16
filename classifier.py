import os, json, re, sys, numpy
from nltk.stem import WordNetLemmatizer
from optparse import OptionParser
from stopword import Stopword
from llda import LLDA

class Classifier:


    def __init__(self, options):

        self.options = options
        self.file_dir = "./build/"
        self.stopwords = Stopword(self.file_dir).get_stopwords()
        self.labels= []
        self.corpus = []

    def train_model(self, filename, model_name):
        self.create_label_corpus(filename)
        llda = LLDA(self.options.K, self.options.alpha, self.options.beta)
        llda.set_corpus(self.labelset, self.corpus, self.labels)
        print "M=%d, V=%d, L=%d, K=%d" % (len(self.corpus), len(llda.vocas), len(self.labelset), self.options.K)
        for index in range(self.options.iteration):
            sys.stderr.write("-- %d : %.4f\n" % (index, llda.perplexity()))
        print "perplexity : %.4f" % llda.perplexity()
        phi = llda.phi()
        theta = llda.theta()
        new_stopword = []
        for k, label in enumerate(self.labelset):
            print "\n-- label %d : %s" % (k, label)
            # for w in numpy.argsort(-phi[k])[:30]:
            for w in numpy.argsort(-phi[k]):
                # if phi[k,w] >= 0.0001:
                print "%s: %f" % (llda.vocas[w], phi[k,w])

    def lemmatize(self, string):
        return WordNetLemmatizer().lemmatize(string, pos='v')

    def update_stopwords(self):
        new_stopword = []
        phi = llda.phi()
        theta = llda.theta()
        for k, label in enumerate(self.labelset):
            for w in numpy.argsort(-phi[k]):
                if phi[k,w] <= 0.001:
                    new_stopword.append(llda.vocas[w])
                else:
                    if llda.vocas[w] in new_stopword:
                        new_stopword.remove(llda.vocas[w])
        Stopword(self.file_dir).update_stopword(new_stopword)

    def create_label_corpus(self,filename):
        with open(os.path.join(self.file_dir,filename)) as model:
            for row in model:
                label_class_list = []
                selected_labels = []

                split_row = row.lower().split("\"|\"")
                label_array = re.sub(r'\W+',' ',split_row[0]).split()
                # Create Unicoded label_type
                for label_type in re.sub(r'\W+',' ',split_row[1]).split():
                    label_class_list.append(unicode(label_type,"utf-8"))

                for label in label_array[:-1]:
                    lemmatized_label = self.lemmatize(label)
                    if label not in self.stopwords and len(label) > 2 and not bool(re.search(r'\d',lemmatized_label)) and lemmatized_label not in self.stopwords:
                        selected_labels.append(lemmatized_label)

                self.corpus.append(selected_labels)
                self.labels.append(label_class_list)
                self.labelset = list(set(reduce(list.__add__, self.labels)))


parser = OptionParser()
parser.add_option("-f", dest="filename", type="string", help="filename", default="stopwords.txt")
parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.0005)
parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.00125)
parser.add_option("-k", dest="K", type="int", help="number of topics", default=50)
parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
parser.add_option("-n", dest="samplesize", type="int", help="dataset sample size", default=100)
(options, args) = parser.parse_args()
classifier = Classifier(options)
classifier.train_model("test.txt","test")
