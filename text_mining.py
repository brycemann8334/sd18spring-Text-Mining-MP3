from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import MDS
from nltk.corpus import stopwords
from os.path import exists
import numpy as np
import matplotlib.pyplot as plt
import string
import doctest
import pickle
import requests

def get_word_str(file_name, begin_str=None, end_str=None):
    r"""Read the specified text file, and isolates the lines between the
    begin_str and end_str. Returns a lowercase str of the file without
    punctuation that can then be vectorized.

    inputs:
    file_name: txt file that you want to analyze

    Normally, you must include "\n" at the end of begin_str and end_str so that
    when it looks for that line, if its the beginning/end of the text it needs
    the newline character to match and isolate the correct part of the file. If
    you don't include a begin_str it will default to the Project Gutenberg
    standard heading and if you don't include an end_str it will continue
    to the end of the txt file.

    UNIT TEST:
    Testing done below since the outputes from putting books in is much
    longer than what I want to copy into a doctest
    """
    f = open(file_name, 'r')
    lines = f.readlines()
    curr_line = 0
    #check for a begin_str
    if begin_str==None:
        #skip over lines that are part of the project Gutenberg legalese
        while lines[curr_line].find('***') == -1:
            curr_line += 1
        lines = lines[curr_line+1:]
    else:
        #same as above but with user entered begin_str
        while lines[curr_line].find(begin_str) == -1:
            curr_line += 1
        lines = lines[curr_line+1:]
    #check if user entered an end_str
    if end_str==None:
        while lines[curr_line].find('*** END OF') == -1:
            curr_line += 1
        lines = lines[:curr_line]
    else:
        #cut out all of the authors notes and footnotes after the story
        while lines[curr_line].find(end_str) == -1:
            curr_line += 1
        lines = lines[:curr_line]

    word_str = ''
    #create lowercase string of all words in the story
    for i in range(len(lines)):
        word_str += lines[i].lower()
    #remove digits from text
    word_str = ''.join([j for j in word_str if not j.isdigit()])
    #use dictionary comprehension to make a map so translate removes punctuation
    table = str.maketrans({key: None for key in string.punctuation})
    word_str = word_str.translate(table)
    #strip all whitespace, replace with spaces
    word_str = word_str.replace("\r", ' ')
    word_str = word_str.replace("\n", ' ')
    word_str = word_str.replace("\t", ' ')

    return word_str

def add_to_data_set(file_name, title_str, begin_str=None, end_str=None):
    """
    creates a word_str and stores it in a list in a file

    takes a title_str and stores it in a dictionary, where the key is the index
    value of the string in the list above, and the value is the title you want
    to associate with that file.

    allows you to build and organize a growing collection of files to analyze.
    If you want to rebuild the data set either move or delete the str and title
    files from the directory

    UNIT TEST:
    Tested towards bottom of page using test file names and test title, printed
    the contents of files to ensure it worked.
    """
    str_file='texts_to_analyze'
    title_file='labels'
    #get the word_str to store
    word_str = get_word_str(file_name, begin_str, end_str)

    #check if both files exist
    if exists(str_file) and exists(title_file):
        #pull out existing file contents
        list_of_str = pickle.load(open(str_file, 'rb+'))
        title_dict = pickle.load(open(title_file, 'rb+'))

        #only adds the new string if it is not already in data set
        if word_str not in list_of_str:
            #adding str and title
            list_of_str.append(word_str)
            title_dict[len(list_of_str)-1]=title_str

            #dump the new structures back into the file
            pickle.dump(list_of_str, open(str_file, 'rb+'), protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(title_dict, open(title_file, 'rb+'), protocol=pickle.HIGHEST_PROTOCOL)

            #if word_str is in list_of_str, nothing is modified to eliminate
            #redundant analysis

    else:
        #if the files don't exist, create the first entry for the structures
        list_of_str = [word_str]
        title_dict = {0: title_str}
        #dump the entry into a new file for storage
        pickle.dump(list_of_str, open(str_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(title_dict, open(title_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def compute_similarity_matrix(file_name, title_str, begin_str=None, end_str=None,
    stopword_str=None):
    """
    input a file_name and a title you want associated with the file.

    if you want analysis to not include most common english words, set 4th
    argument to 'english'

    UNIT TEST:
    Tested at the bottom of the page by calling add_to_data_set for test1,
    then compute_similarity_matrix with test2
    """
    #adds new file to data set
    add_to_data_set(file_name, title_str, begin_str, end_str)
    #load str_list
    str_list = pickle.load(open('texts_to_analyze', 'rb'))

    #check if you want to ignore the use of some common words
    #look at nltk documentation to find what types of stopword_strs there are
    #set up the vectorizer either w/ or w/o words to ignore
    if stopword_str==None:
        tfidf_vectorizer = TfidfVectorizer()
    else:
        words_to_ignore = stopwords.words('english')
        tfidf_vectorizer = TfidfVectorizer(stop_words=words_to_ignore)

    #pass list of strings to the vectorizer
    similarity_matrix = tfidf_vectorizer.fit_transform(str_list)

    #return the cosine similarity matrix from the vectorized list
    return cosine_similarity(similarity_matrix)

def plot_similarity_cluster(cosine_matrix):
    """
    given the computed similarity matrix and the file filled with user inputted
    titles, plot and label the similarity cluster.

    most of the code was given on the soft des website, and I spent a while
    making sure I understood what the coords were and designing my method of
    labeling in a way that worked with the coordinate matrix

    UNIT TEST:
    Tested at the bottom of the page by adding 2 files to compare to test1
    and making sure labeling/clustering was working well
    """
    #load title dictionary for labeling points
    title_dict = pickle.load(open('labels', 'rb'))

    #get dissimilarity
    dissimilarity = 1 - np.asarray(cosine_matrix)
    # compute the embedding
    coord = MDS(dissimilarity='precomputed').fit_transform(dissimilarity)
    print(coord)
    #plot values in the first column against the corresponding value in the
    #second column
    plt.scatter(coord[:,0], coord[:,1])
    #coord.shape[0] gives the number of rows, which correspond to the added files
    for i in range(coord.shape[0]):
        #label using the title that was entered when the file was added to set
        plt.annotate(title_dict[i], (coord[i,:]))

    plt.show()

def get_user_input():
    """
    returns all of the user input that run_similarity_calculator needs
    """
    file_name = input('Enter the name of the text file (including .txt) that ' +
        'you want to add to the analysis: ')
    title_str = input('Enter the label that you want this text to have when ' +
        'it is plotted: ')
    begin_str = input('Enter the words in the text file where you want to start ' +
        'analyzing (optional, only enter something if the file is not from ' +
        'Project Gutenberg): ')
    if begin_str=='':
        begin_str=None
    end_str = input('Enter the line in the text file where you want to ' +
        'end analyzing the text (optional, only enter something if the file is not from ' +
        'Project Gutenberg): ')
    if end_str=='':
        end_str=None
    stpwrd = input('Enter [Y] if you want to include common english words ' +
        'in the analysis or [N] if you do not want to include those words: ')
    if stpwrd == 'N':
        stopword_str = 'english'
    else:
        stopword_str = None

    #print(file_name, str_file, title_str, title_file, begin_str, end_str, stopword_str)
    return file_name, title_str, begin_str, end_str, stopword_str

def run_similarity_calculator(file_name, title_str, begin_str, end_str,
    stopword_str):
    """consolidates user input and runs the program"""
    cosine_matrix = compute_similarity_matrix(file_name, title_str, begin_str,
        end_str, stopword_str)

    plot_similarity_cluster(cosine_matrix)

def add_source():
    """prompts the user to add another source to the folder"""

    url = input('Please paste a link to a file that you want to download ' +
            'as a txt file to be analyzed later: ')
    file_name = input('Please type what you want to save this file as ' +
            '(include the .txt extension): ')
    page = requests.get(url).text
    f = open(file_name, 'w')
    f.write(page)
    f.close()
    print('Your file has been saved to the folder...')

if __name__=='__main__':
    while True:
        choice = input('Enter the number [0] if you would like to add another '+
            'source to analyze, or the number [1] to run the similarity calculator: ')

        if choice==str(0):
            add_source()
        elif choice==str(1):
            file_name, title_str, begin_str, end_str, stopword_str = get_user_input()
            run_similarity_calculator(file_name, title_str, begin_str, end_str,
                stopword_str)
        else:
            print('That was not a choice, please choose either [0] or [1].')
            continue

        continue_running = input('Enter [Y] if you would like to add or analyze ' +
            'more sources or [N] if you would rather come back later: ')
        if continue_running=='N':
            break


"""SOME OF THE ARGUMENTS BELOW MAY BE INCORRECT NOW AS I CHANGED SOME OF THE
    FUNCTIONS SO THAT IT WOULD BE CLEANER"""
#                            testing get_word_str
    #print(get_word_str('huck_finn.txt')[:10000])
    #print(get_word_str('huck_finn.txt')[-10000:])

#                            testing add_to_data_set
#add_to_data_set('test1.txt','test_title1', 'Beginning.\n', 'End.\n')
#add_to_data_set('test2.txt','test_title2', 'Beginning.\n', 'End.\n')


#                            testing compute_similarity_matrix
#add_to_data_set('test1.txt', 'test_str_file.txt', 'test_title1',
    #'test_title_file.txt', 'Beginning.\n', 'End.\n')
#print(compute_similarity_matrix('test2.txt','test_title2', 'Beginning.\n',
    #'End.\n',))

#                            testing plot_similarity_cluster
#add_to_data_set('test1.txt', 'test_title1', 'Beginning.\n', 'End.\n')
#plot_similarity_cluster(compute_similarity_matrix('test2.txt', 'test_title2',
    #'Beginning.\n', 'End.\n'),
#plot_similarity_cluster(compute_similarity_matrix('illiad.txt', 'illiad',
    #'test_title_file.txt'), 'test_title_file.txt')
