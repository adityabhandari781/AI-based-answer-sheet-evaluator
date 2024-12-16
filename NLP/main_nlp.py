# IMPORTS
from gramformer import Gramformer
gf = Gramformer(models=1, use_gpu=False)
from transformers import pipeline
import spacy
nlp = spacy.load("en_core_web_sm")
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')



# CONTEXT CORRECTION
def correct_grammar(text):
    return list(gf.correct(text, max_candidates=1))[0]



# SPELLING CORRECTION (only for answers with few words)
fix_spelling = pipeline("text2text-generation",model="oliverguhr/spelling-correction-english-base")

def correct_spelling(text):
    return fix_spelling(f"{text}",max_length=2048)[0]['generated_text']



# SYNONYM SIMILARITY
def generate_synonyms(word):
    synonyms = []
    for token in nlp.vocab:
        if token.has_vector and token.is_alpha and token.text.lower() != word.lower():
            similarity = token.similarity(nlp(word)[0])
            if similarity > 0.7:  # Threshold for synonym similarity
                synonyms.append(token.text)
    return synonyms

def calculate_synonym_similarity(true_answer, student_answer):
    true_tokens = [token.text.lower() for token in nlp(true_answer) if token.is_alpha]
    student_tokens = [token.text.lower() for token in nlp(student_answer) if token.is_alpha]

    match_count = 0

    for student_word in student_tokens:
        if student_word in true_tokens:
            match_count += 1
        else:
            synonyms = generate_synonyms(student_word)
            match_count += sum(1 for synonym in synonyms if synonym in true_tokens)

    avg_length = (len(true_tokens) + len(student_tokens)) / 2
    synonym_similarity = match_count / avg_length if avg_length > 0 else 0
    return synonym_similarity



# BIGRAM SIMILARITY
def generate_bigrams(word_list):
    return [(word_list[i], word_list[i + 1]) for i in range(len(word_list) - 1)]

def bigram_similarity(text1, text2):
    list1 = text1.split()
    list2 = text2.split()
    bigrams1 = generate_bigrams(list1)
    bigrams2 = generate_bigrams(list2)
    
    set_bigrams1 = set(bigrams1)
    set_bigrams2 = set(bigrams2)
    
    common_bigrams = set_bigrams1.intersection(set_bigrams2)
    common_count = len(common_bigrams)
    
    avg_bigram_length = (len(set_bigrams1) + len(set_bigrams2)) / 2.0
    
    similarity_score = common_count / avg_bigram_length if avg_bigram_length > 0 else 0    
    return similarity_score



# COSINE SIMILARITY
def cos_similarity(sentence1, sentence2):
    embedding1 = model.encode(sentence1, show_progress_bar=False)
    embedding2 = model.encode(sentence2, show_progress_bar=False)
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity



# OVERALL SENTENCE SIMILARITY
def avg_similarity(text1, text2):
    return 0.1 * bigram_similarity(text1, text2) + 0.2 * calculate_synonym_similarity(text1, text2) + 0.7 * cos_similarity(text1, text2)



# TEXT PREPROCESSING
def remove_stopwords(text):
    return " ".join([word.lower() for word in text.split() if word.lower() not in nlp.Defaults.stop_words])
def lemmatize_text(text):
    return " ".join([word.lemma_ for word in nlp(text)])
def preprocess_text(text):
    text = lemmatize_text(text)
    text = remove_stopwords(text)
    return text



# PARAGRAPH SIMILARITY
def cluster_sentences(sentences, n):
    m = len(sentences)
    if m <= n:
        # If fewer sentences than clusters, each sentence is a cluster
        return [[sentence] for sentence in sentences]

    # Calculate similarities between consecutive sentences
    similarities = [avg_similarity(sentences[i], sentences[i + 1]) for i in range(m - 1)]

    # Find indices of the (m - n) smallest similarities
    partition_indices = sorted(range(len(similarities)), key=lambda i: similarities[i])[:n - 1]

    # Sort partition indices to split sentences sequentially
    partition_indices.sort()

    # Partition sentences based on the identified indices
    clusters = []
    prev_index = 0
    for idx in partition_indices:
        clusters.append(sentences[prev_index:idx + 1])
        prev_index = idx + 1
    clusters.append(sentences[prev_index:])  # Add the final cluster

    return clusters

def compare_answers(student_answer, answer_key, max_marks):
    
    # student_answer = nlp(preprocess_text(student_answer))
    # answer_key = nlp(preprocess_text(answer_key))
    
    # Get the sentences from the student answer and the answer key
    student_sentences = [str(i) for i in student_answer.sents]
    answer_key_sentences = [str(i) for i in answer_key.sents]
    
    student_clusters = cluster_sentences(student_sentences, max_marks)
    key_clusters = cluster_sentences(answer_key_sentences, max_marks)
    # for i in student_clusters:
    #     print(i)
    # for i in key_clusters:
    #     print(i)
    
    total_marks = 0
    confidence = []

    # Compare student clusters with key clusters
    for student_cluster in student_clusters:
        similarity_dict = {}        
        for student_sentence in student_cluster:
            for key_cluster in key_clusters:
                for key_sentence in key_cluster:
                    similarity = avg_similarity(student_sentence, key_sentence)
                    similarity_dict[key_sentence] = similarity
        
        # Find the key sentence with the maximum similarity
        max_pair = max(similarity_dict, key=similarity_dict.get)
        max_key_sentence = max_pair
        
        # Find the cluster associated with the key sentence
        key_cluster = next(cluster for cluster in key_clusters if max_key_sentence in cluster)
        
        # Calculate the average similarity between clusters
        cluster_similarities = []
        for student_sentence in student_cluster:
            for key_sentence in key_cluster:
                cluster_similarities.append(avg_similarity(student_sentence, key_sentence))
        
        average_similarity = sum(cluster_similarities) / len(cluster_similarities)
        
        # Add 1 mark if average similarity > 0.5
        # print(average_similarity)
        confidence.append(average_similarity)
        if average_similarity > 0.3:
            total_marks += 1
        
        # Remove the used key cluster
        key_clusters.remove(key_cluster)
    
    return total_marks, confidence



# MAIN FUNCTIONS

def evaluate_sentence(student_answer, answer_key):
    # Correct spelling in student answer
    student_answer = correct_spelling(student_answer)
    confidence = avg_similarity(student_answer, answer_key) # no pre-processing for single sentences
    marks = round(confidence)
    return marks, confidence

def evaluate_paragraph(student_answer, answer_key, max_marks):
    # # Correct spelling in student answer
    # student_answer = correct_spelling(student_answer)

    # Correct grammar in student answer
    student_answer = correct_grammar(student_answer)
    
    # Compare student answer with answer key
    student_answer = nlp(preprocess_text(student_answer))
    answer_key = nlp(preprocess_text(answer_key))
    
    marks, confidence = compare_answers(student_answer, answer_key, max_marks)
    return marks, confidence