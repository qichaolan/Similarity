'''
    @author: Qichao Lan

'''
import sys
import json
import argparse
import gensim
import itertools
import uuid
from genericpath import isdir
from multiprocessing import cpu_count
from os import walk, path, sep
from time import time
from collections import defaultdict

class JavaSimilarity(object):
    def __init__(self):
        pass

    def __get_file_list_by_ext(self, in_dir, ext):
        '''
            search for training candidate files by a gaving extension
        '''
        files = {}
        idx = 0
        
        if isdir(in_dir):
            for dirpath, dirnames, filenames in walk(in_dir):
                for filename in filenames:
                    # check if file extension matches
                    if "." not in filename: continue
                    if filename.split(".")[-1].lower() != ext.lower(): continue
                    # append current file
                    files[idx] = path.join(dirpath, filename)
                    idx+=1
        return files

    def __read_tokens(self, file):
        '''
            read a list of files line by line, and pre-process each line using gesim

            return a list of words
        '''
        with open(file, 'r') as f:
            for line in f:
                tokens = gensim.utils.simple_preprocess(line)
                if len (tokens) < 1: continue
                yield tokens
        
    def __is_ascii(self, s):
        return all(ord(c) < 128 for c in s)

    def __build_corpus_from_java_file(self, java_file):
        '''
            build corpus for java files
        '''

        # Here is a list of keywords in the Java programming language. 
        stoplist = set('''abstract continue for new switch assert*** default goto* package synchronized
        boolean do if private this
        break double implements protected throw
        byte else import public throws
        case enum**** instanceof return transient
        catch extends int short try
        char final interface static void
        class finally long strictfp** volatile
        copyright inc licensed
        const* float native super while'''.split())

        # build tokens
        codes = list(self.__read_tokens(java_file))

        # remove java keywords, and non-ASCII string.
        texts = [
            [word for word in code if word not in stoplist and self.__is_ascii(word)]
            for code in codes
        ]

        return list(itertools.chain(*texts))

    def __build_corpus_from_java_libs(self, input_dir):
        '''
            build model for java files
        '''
        # read java files from input dir
        java_files = self.__get_file_list_by_ext(input_dir, "java")

        # build corpus from the input dir 
        texts = [
                self.__build_corpus_from_java_file(java_files[idx])
                    for idx in sorted(java_files, reverse=False)
        ]

        # remove words that appear only once
        frequency = defaultdict(int)

        for text in texts: 
            for token in text:
                frequency[token] += 1

        texts = [
            [token for token in text if frequency[token] > 1]
            for text in texts
        ]

        return texts, java_files

    def __build_corpus_from_a_java_lib(self, input_file):
        '''
            build model for java files
        '''
        # build corpus from the input dir 
        texts = self.__build_corpus_from_java_file(input_file)
        # remove words that appear only once
        frequency = defaultdict(int)
        for token in texts:
            frequency[token] += 1

        texts = [ token for token in texts if frequency[token] > 1]
        
        return texts

    def __get_model_file_names(self, dir):
        __uuid = str(uuid.uuid1())
        return path.join(dir, __uuid+".index"), path.join(dir, __uuid+".dct"), path.join(dir, __uuid+".lsi"), path.join(dir, __uuid+".json")

    def __save_file_list_to_json(self, dct, json_file):
        with open(json_file, 'w') as f:
                json.dump(dct, f)

    def json_to_dict(self, filename):
        with open(filename) as f_in:
            return json.load(f_in)

    def BuildSimilarity(self, train_data, output_dir):
        '''
            To prepare for similarity queries, 
            
            We need to enter all documents which we want to compare against subsequent queries.

            All the input data will be used for training LSI, converted to 2-D LSA space. 

            As an estimation: a corpus of one million documents would require 2GB of RAM in a 256-dimensional LSI space
        
        '''
        idx_file, dct_file, lsi_file, json_file = self.__get_model_file_names(output_dir)

        # Creating the Dictionary
        texts, java_lst = self.__build_corpus_from_java_libs(train_data)
        dct = gensim.corpora.Dictionary(texts)
        dct.save_as_text(dct_file)

        # saving file list
        self.__save_file_list_to_json(java_lst, json_file)
        
        # Creating the Corpus
        corpus = [dct.doc2bow(text) for text in texts]
        # Indexing by Latent Semantic Analysis" <http://www.cs.bham.ac.uk/~pxt/IDA/lsa_ind.pdf>`_
        lsi = gensim.models.LsiModel(corpus, id2word=dct, num_topics=2)
        lsi.save(lsi_file)

        # Initializing query structures
        # transform corpus to LSI space and index it
        index = gensim.similarities.MatrixSimilarity(lsi[corpus],num_best = 5)  
        index.save(idx_file)
        
        return idx_file, dct_file, lsi_file, json_file

    def QuerySimilarity(self, idx_file, dct_file, lsi_file, name_map, test_data):
        '''
            Query the top five mostsimilar documents from a pre-trainned Similarity
        
        '''
        # convert the test data to LSI space
        texts = self.__build_corpus_from_a_java_lib(test_data)
        dct = gensim.corpora.Dictionary.load_from_text(dct_file)
        vec_bow = dct.doc2bow(texts)

        # load the model file, and return only the top five most similar codes
        lsi = gensim.models.LsiModel.load(lsi_file)
        index = gensim.similarities.MatrixSimilarity.load(idx_file)
        
        #  perform a similarity query against the corpus
        vec_lsi = lsi[vec_bow]  
        sims = index[vec_lsi] 
        
        # return a sorted (document_number, document_similarity) 2-tuples
        return sorted(enumerate(sims), key=lambda item: item[1], reverse = False)

    @staticmethod
    def main(argv):
        '''
         the main method that actutally runs the entirety of the similarity comparsion  
        '''
        parser = argparse.ArgumentParser(description='Process some integers.')

        parser.add_argument("-t", "--train", type=str, required=True,
            help="Path to a directory which store the training data")

        parser.add_argument("-e", "--evaluating", type=str, required=True,
            help="Path to a directory which store the test data")

        parser.add_argument("-o", "--output", type=str, required=True,
            help="Path to a directory which store the model data")

        start_time = time()

        args = parser.parse_args()

        if not isdir(args.train):
            print("Invaild input dir! Aborting.")
            exit(-1)

        if not path.exists(args.evaluating):
            print("Invaild test file! Aborting.")
            exit(-1)

        if not isdir(args.output):
            print("Invaild output dir! Aborting.")
            exit(-1)

        # creation and initialization  
        jSimilarity = JavaSimilarity()

        # train the model
        idx_file, dct_file, lsi_file, name_map = jSimilarity.BuildSimilarity(args.train, args.output)
        print ("Index file: ", idx_file)
        print ("Dictionary file: ", dct_file)
        print ("Lsi model file: ", lsi_file)
        print ("File mapping: ", name_map)

        nmap = jSimilarity.json_to_dict(name_map)
        # test the model
        sims = jSimilarity.QuerySimilarity(idx_file, 
            dct_file, 
            lsi_file, 
            name_map, 
            args.evaluating)

        #printing summary
        print ("\n########################################################\n\n")
        print ("Evaluating the file: ", args.evaluating)
        print ("\nThe Top Five Most Similar Codes Are:\n")
        for doc_position, doc_score in sims:
            print(nmap[str(doc_score[0])], "Similarity Score: ", doc_score[1])
        print ("\n########################################################\n\n")
        print ("Job done: whole process use ", time() - start_time, " s!!!!")

        return

if __name__ == '__main__':
    exit(JavaSimilarity.main(sys.argv[1:]))