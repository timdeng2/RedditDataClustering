import random
import pandas as pd
from sklearn import feature_extraction
from sklearn.cluster import KMeans
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
import spacy
nlp = spacy.load("en_core_web_sm")
#--------------------------------------------------------------------------------
##Define a function to run the clustering so we can iterate
def cluster(stopwords, data_df, cluster_prefix = "Main"):

    #Now make a new vectorizer with those stopwords excluded; this version will use TF-IDF feature weighting
    features = feature_extraction.text.TfidfVectorizer(input='content', 
                                                    encoding='utf-8', 
                                                    decode_error='ignore', 
                                                    lowercase=True,
                                                    stop_words = stopwords,
                                                    tokenizer = None,
                                                    ngram_range=(1, 1), 
                                                    analyzer='word', 
                                                    max_features=10000,   #Larger vocabulary for clustering
                                                    )

    #Sklearn first fits then transforms
    features.fit(data_df.loc[:,"title"].values)

    #Now extract the content features (with phrases, without stopwords)
    x = features.transform(data_df.loc[:,"title"].values)
    print(x)
    print(x.shape)

    #Cluster documents by content
    cluster = KMeans(n_clusters = 10,       #The number of topics we'll get
                        init = "k-means++", 
                        n_init = "auto", 
                        max_iter = 30000, 
                        algorithm = "lloyd",
                        )

    #Perform clustering
    cluster.fit(x)

    #Add topic labels to dataframe
    data_df.loc[:,"Topic"] = [cluster_prefix+"_"+str(x) for x in cluster.labels_]

    #Sort by topic
    data_df.sort_values("Topic", inplace = True)
    print(data_df)
    print(data_df.value_counts("Topic"))

    #Get the biggest topic for further splitting
    most_frequent = data_df.value_counts("Topic").index[0]
    most_frequent_count = data_df.value_counts("Topic").iloc[0]
    print("Most frequent: ", most_frequent)

    #Separate the main topic from other topics
    main_topic = data_df[data_df.loc[:,"Topic"] == most_frequent]
    other_topics = data_df[data_df.loc[:,"Topic"] != most_frequent]

    #Send back the two dataframes
    return main_topic, other_topics, most_frequent
#--------------------------------------------------------------------------------
def extract_syntactical_features(text):
    return ' '.join([token.pos_ for token in nlp(text)])


def cluster2(data_df, cluster_prefix = "Main"): #cluster by syntax
    syntactical_features = [extract_syntactical_features(text) for text in data_df.loc[:,"title"].values]


    vectorizer = feature_extraction.text.CountVectorizer()
    X = vectorizer.fit_transform(syntactical_features)
    print("finished POS")

    #Cluster documents by content
    cluster = KMeans(n_clusters = 5,       #The number of topics we'll get
                        init = "k-means++", 
                        n_init = "auto", 
                        max_iter = 30000, 
                        algorithm = "lloyd",
                        )
    cluster.fit(X)
    print(cluster.labels_)

    data_df.loc[:,"Syntax"] = [cluster_prefix+"_"+str(x) for x in cluster.labels_]

    #Sort by topic
    data_df.sort_values("Syntax", inplace = True)
    print(data_df)
    print(data_df.value_counts("Syntax"))

    #Get the biggest topic for further splitting
    most_frequent = data_df.value_counts("Syntax").index[0]
    most_frequent_count = data_df.value_counts("Syntax").iloc[0]
    print("Most frequent: ", most_frequent)

    #Separate the main topic from other topics
    main_topic = data_df[data_df.loc[:,"Syntax"] == most_frequent]
    other_topics = data_df[data_df.loc[:,"Syntax"] != most_frequent]

    #Send back the two dataframes
    return main_topic, other_topics, most_frequent




# #Load the corpus
# file = "News.NYT.1931-1969.gz"
# data_df = pd.read_csv(file, index_col = 0)
# print(data_df)

# #Choose a random year
# years = list(set(data_df.loc[:,"Year"].values))

# #Reduce data to only one year
# year = random.choice(years)
# data_df = data_df[data_df.loc[:,"Year"] == year]
# print(data_df)

# #Use association measures to find multi-word expressions in Gensim
# phrase_model = Phrases([doc.split() for doc in data_df.loc[:,"Text"].values], 
#                         min_count = 2, 
#                         threshold = 0.7, 
#                         connector_words = ENGLISH_CONNECTOR_WORDS, scoring = "npmi"
#                         )

# print(phrase_model.export_phrases().keys())
# print("ABOVE: Learned phrases")

# #Replace phrases in the df
# data_df.loc[:,"Text"] = [" ".join(phrase_model[sentence.split()]) for sentence in data_df.loc[:,"Text"]]

# #First find the most frequent words
# features = feature_extraction.text.CountVectorizer(input='content', 
#                                                 encoding='utf-8', 
#                                                 decode_error='ignore', 
#                                                 lowercase=True, 
#                                                 tokenizer = None,
#                                                 ngram_range=(1, 1), 
#                                                 analyzer='word', 
#                                                 max_features=500,   #Choose number of future stopwords
#                                                 )

# #Sklearn first fits then transforms
# features.fit(data_df.loc[:,"Text"].values)
# #The most frequent words can be found in the dictionary of vocabulary items
# stopwords = list(features.vocabulary_.keys())
# print(stopwords)
# print("ABOVE: Frequent words to exclude")

# #Create a loop to continue clustering until the largest category is not too big
# main_topic = data_df    #Initialize main topic
# cluster_prefix = "Topic"     #Start with root topics
# holder = []
# starting_length = len(data_df)
# counter = 0

# while True:

#     #Run clustering
#     counter += 1
#     main_topic, other_topics, most_frequent = cluster(stopwords, main_topic, cluster_prefix)
#     cluster_prefix = str(most_frequent)

#     #Check stopping conditions, no topic over 20% of documents
#     if len(main_topic)/len(data_df) < 0.20:
#         holder.append(other_topics)
#         holder.append(main_topic)
#         break

#     #Keep going
#     else:
#         holder.append(other_topics)
#         print("Continuing after round " + str(counter), "Current: ", len(main_topic), "Total: ", starting_length)
        

#Merge
# data_df = pd.concat(holder)
# data_df.sort_values("Topic", inplace = True)
# #Reorder columns
# data_df = data_df.loc[:,["Year", "Month", "Day", "Topic", "Text"]]
# print(data_df)
# print(data_df.value_counts("Topic"))

# #Save
# data_df.to_csv("Test."+str(year)+".csv")