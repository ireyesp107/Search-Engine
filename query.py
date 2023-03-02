import file_io
import sys
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

class query:
    #boolean then as strings
    def __init__(self ):
        self.id_pagerank: dict = {}
        self.id_title: dict = {}
        self.word_id_relv: dict = {}

        self.STOP_WORDS = set(stopwords.words('english'))
        self.nltk_test = PorterStemmer() 

        self.read()

    def read(self):
        file_io.read_title_file(sys.argv[-3], self.id_title)
        file_io.read_docs_file(sys.argv[-2], self.id_pagerank)
        file_io.read_words_file(sys.argv[-1], self.word_id_relv)

        user_input = ""
        while True:
            user_input = input("input search! (type \":quit\" to finish): ")
            if user_input == ":quit":
                sys.exit(1)
            all_query_words = re.findall('''\[\[[^\[]+?\]\]|[a-zA-Z0-9]+'[a-zA-Z0-9]+|[a-zA-Z0-9]+''', user_input)
            query_mod_words = []
            for word in all_query_words:
                word = self.stem_remove_stops(word)
                if word != None:
                    query_mod_words.append(word)
            self.handle_query(query_mod_words)

    def stem_remove_stops(self, word: str):
        if word not in self.STOP_WORDS:
            word = self.nltk_test.stem(word)
            word = word.lower()
            return word
        return None
    
    def handle_query(self, words: list):
        id_score: dict = {}
        for word in words:
            if word in self.word_id_relv.keys():
                for id in self.word_id_relv[word]:
                    if id not in id_score.keys():
                        id_score[id] = self.word_id_relv[word][id]
                    else:
                        id_score[id] += self.word_id_relv[word][id]
        
        # add pagerank to the score if specified
        if sys.argv[-4] == "--pagerank":
            for id in id_score:
                id_score[id] = id_score[id] * self.id_pagerank[id]
        
        # making dict into list of tuples
        sorted_ids = []
        for id in id_score:
            sorted_ids.append((id, id_score[id]))

        # sort and print list of tuples
        sorted_ids.sort(reverse=True, key=self.sort_tuples)
        if len(sorted_ids) >= 10:
            print("Here are the top 10 results:")
            for el in range (0,10):
                title = self.id_title[sorted_ids[el][0]]
                print(str(el+1)+". "+title)
        else:
            print("Here are all the results:")
            counter = 1
            for el in sorted_ids:
                title = self.id_title[el[0]]
                print(str(counter)+". "+title)
                counter += 1
    
    def sort_tuples(self, tup):
        return tup[1]

if __name__ == "__main__":
    query().main()
    
# python3 query.py "Titles.txt" "Documents.txt" "Words.txt"
# python3 query.py --pagerank "Titles.txt" "Documents.txt" "Words.txt"