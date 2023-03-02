from cmath import inf
import math
from tokenize import String
import xml.etree.ElementTree as et
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pyparsing import identbodychars
import file_io
import sys

class index:
    def __init__(self, wiki:str, title:str, doc:str, word:str):
       """
       Constructor initlialized all dictionaries
       Argu file:str= reads in file
       """
       self.wiki_xml_root= et.parse(wiki).getroot()
       self.all_pages: et = self.wiki_xml_root.findall("page")
       self.stop_words=set(stopwords.words('english'))
       self.nltk_test = PorterStemmer()
       self.word_id_count={}
       self.ids_to_titles={}
       self.title_to_id={}
       self.id_to_links={}
       self.freq={}
       self.freq_inv={}
       self.rel={}
       self.weight={}
       self.n_regex = '''\[\[[^\[]+?\]\]|[a-zA-Z0-9]+'[a-zA-Z0-9]+|[a-zA-Z0-9]+'''
       self.num_of_pages: int = len(self.all_pages)
       self.id_to_pagerank = {}
       self.id_word_count={}
       #code fills in our dictionaries
       self.process()
       self.term_freq()
       self.inv_term_freq()
       self.calc_term_rel()
       self.fill_id_to_link()
       self.calc_weight()
       self.calc_pagerank()
 
       file_io.write_title_file(title, self.ids_to_titles) #write files to be accessed in query
       file_io.write_docs_file(doc, self.id_to_pagerank) #write files to be accessed in query
       file_io.write_words_file(word, self.rel)
    
    def process(self):
        """
        process parses through the words within the doc and checking whether it is a link
        The function goes through all the pages and fills our dictionaries ids_to_titles, title_to_id,
        self.id_word_count, word_id_count.
        Parameter:Self lets us use the variabls defined in the constructor

        return 
        

        """
        for page in self.all_pages:
            title: str = page.find("title").text.strip()
            id: int = int(page.find("id").text)
            self.ids_to_titles[id] = title
            self.title_to_id[title] = id
            self.id_word_count[id] = {}

            # next make word corpus
            words: str = page.find("text").text
            all_words = re.findall(self.n_regex, words)

            for word in all_words:
                if "[[" and "]]" in word:
                    word = word[2:-2]
                    #calls the helper method which returns the all the words to the right of the column
                    if "|" in word:
                        word = self.handle_vert_line(word)
                    #calls helper method which removes the colon and treates all of the words as words not links
                    if ":" in word:
                        word = self.handle_colon(word)        
                    
                    words_in_link = re.findall(self.n_regex, word)
                    self.stem_remove_stops_list(words_in_link, id) #helper method which removes the stopwords when there are multiple words
                else:
                    self.stem_remove_stops(word, id) #helper method which removes the stop words of a single word
    
    def handle_vert_line(self, link: str):
        """
        crreates a list of words split on the line
        Arguemnt str: takes in words with brackets and a line through the middle
        returns: Strigns to the right of the line
        """
        # get rid of | and everything before it
        link = link.split("|")
        return link[1]

    def handle_colon(self, words: str):
        """
        Method treats every word in the bracket like a real word not a line
        Argument str: Is the string inside the brackets
        Returns the left and right side of the colon
        """
        half_link = words.split(":")
        return half_link[0] + " " + half_link[1]

    def stem_remove_stops(self, word: str, id: int): 
        """
        Method to see if word is in the stop words, if not then it will call
        the helper method to add to the dictionaries
        Argu: words
        Return N/A 

        """
        if word not in self.stop_words:
            word = self.nltk_test.stem(word).lower()
            self.add_to_wordcount(word, id)

    def stem_remove_stops_list(self, words: list, id: int): 
        """
        Cheks if the words are stopwords by calling the helper method
        Argu: self
        Argument: id of the page the word is found
        """
        for word in words:
            word = self.stem_remove_stops(word, id)


    def add_to_wordcount(self, word: str, id: int):
        """
        Will add to our dictionary word_id_count and id_word_count if word is not a stopword
        Argu:self
        Argu:word= the word being being counted
        Argu:id = the page of where thr word is found
        """
        # create empty dict if word not already in wordcount
        if word not in self.word_id_count:
            self.word_id_count[word]={}
        
        # adds 1 to the id in the specific word
        if id not in self.word_id_count[word].keys():
            self.word_id_count[word][id]=1
        elif id in self.word_id_count[word].keys():
            self.word_id_count[word][id]+=1

        # add to id_word_count
        if word not in self.id_word_count[id].keys():
            self.id_word_count[id][word] = 1
        else:
            self.id_word_count[id][word] += 1 

    def term_freq(self):
        """
        Calculates the frequency of each term term by finding the word that occcurs the most frequently
        in a given term and then dividing each term by the count

        Return N/A
        """
        aj_dict = {}
        for id in self.ids_to_titles.keys():
            if len(self.id_word_count[id]) == 0:
                aj_dict[id] = 0
            else:
                aj_dict[id] = max(self.id_word_count[id].values())
        for word in self.word_id_count.keys():
            for id in self.word_id_count[word]:
                cij=self.word_id_count[word][id]
                tf=cij/aj_dict[id]
                if word in self.freq.keys():
                    self.freq[word][id]=tf
                else:
                    self.freq.update({word:{id:tf}}) 

    def inv_term_freq(self):
        """
        The method goes through each word and calculates the word that appears in most docs
        The method then finds the log of this value/pagecount of each word
        Argu:self
        return N/A
        """
        for word in self.word_id_count:
            self.freq_inv.update({word:len(self.word_id_count[word].keys())})

        n: int= max(self.freq_inv.values())
        for word in self.freq_inv:
            self.freq_inv[word]=math.log(n/self.freq_inv[word])

    def calc_term_rel(self):
        """
        Method goes through each word and page and multiples both 
        the freq and invFreq. The calculated value is then stored in the dictionary
        later to be used in our query

        """
        for word in self.freq:
            for id in self.freq[word].keys():
                rel_term: int=(self.freq[word][id]*self.freq_inv[word])
                if word in self.rel:
                    self.rel[word][id]=rel_term
                else:
                    self.rel.update({word:{id:rel_term}})
    

    def calc_pagerank(self):
        """
        We use a helper distance method and create our wieghts before this method.
        The method followed the page rank algorithm. Pagerank checks to see how authoirtaive
        a page is by the given number of pages. Then it checks how authoritative the pages
        linked to itself are. Additionally pagerank checks to see if there are fewer links to linked pages
        because it will classify it as more authoritative. Lastly page ranks checks to see
        the influence of K's score by checking how far it is to get from J to K

        Fill fill in the self.id_to_pagerank which is used in Query
        return N/A
        """
        r = {}
        r_prime = {}
        all_ids = self.ids_to_titles.keys()
        for id in all_ids:
            r[id] = 0
            r_prime[id] = 1/self.num_of_pages
        while self.distance(r, r_prime) > 0.001:
            
            print(sum)
            for id in all_ids:
                r[id] = r_prime[id]
            for j in all_ids:
                r_prime[j] = 0
                for k in all_ids:
                    r_prime[j] = r_prime[j] + self.weight[k][j] * r[k]
        self.id_to_pagerank=r_prime
        
        

    
    def distance(self, r: dict, r_prime: dict) -> float:
        """
        Uses a distance forumla to calculate the distance between r_prime and r

        Argu:r= dictionray of r values
        Argu:r_prine= dictationy of r prime values
        
        return the sum of the calculated distance of the entire dictionary of both r_prime and r
        """
        diff_squared = []
        for id in r.keys():
            diff_squared.append((r_prime[id] - r[id]) ** 2)
        return math.sqrt(sum(diff_squared))

    def fill_id_to_link(self):
        """
        Helper Method for calc_weight because it maps an ID to its outgoing links
        Fills up the self.id_to_links 
        """
        for source_page in self.all_pages:
            words: str = source_page.find("text").text
            source_title: str = source_page.find("title").text
            source_id: int = int(source_page.find("id").text)
            all_words = re.findall(self.n_regex, words)
            self.id_to_links[source_id] = set()

            for word in all_words:
                if "[[" and "]]" in word:
                    word = word[2:-2]
                    if "|" in word:
                        link_before_line = word.split("|")
                        word = link_before_line[0]
                    if source_title != word and word in self.title_to_id.keys():  
                        self.id_to_links[source_id].add(self.title_to_id[word])

    def calc_weight(self):
        """
        Our method through special cases in which links not our corpus are ignored
        links from a page to itself are ignored, if a link doesn't link to anything then we
        say it links to everywhere except itself and we check if the link is referenced
        multiple times. Otherwise it will calculate weight based on outgoing pages and the provided formula

        """
        for source_id in self.ids_to_titles.keys():
            self.weight[source_id] = {}
            for target_id in self.ids_to_titles.keys():
                if len(self.id_to_links[source_id]) == 0: # edge case where page links to nothing
                    if target_id != source_id: 
                        self.weight[source_id][target_id] = (0.15/self.num_of_pages) + 0.85*(1/(self.num_of_pages-1))
                    else: 
                        self.weight[source_id][target_id] = (0.15/self.num_of_pages)
                elif target_id in self.id_to_links[source_id]:
                    n_k = len(self.id_to_links[source_id])
                    self.weight[source_id][target_id] = (0.15/self.num_of_pages) + 0.85*(1/n_k)
                else:
                    self.weight[source_id][target_id] = (0.15/self.num_of_pages)

if __name__=="__main__":
    words = sys.argv[0]
    xml_fp=sys.argv[1]
    titles_fp=sys.argv[2]
    docs_fp=sys.argv[3]
    words_fp=sys.argv[4]

    index(xml_fp,titles_fp,docs_fp,words_fp)


#UnitedStates