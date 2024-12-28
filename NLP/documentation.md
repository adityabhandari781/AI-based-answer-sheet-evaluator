## **Table of Contents**
1. [Imports](#1-imports)
2. [Modules and Libraries](#2-modules-and-libraries)
3. [Functions](#3-functions)
   - [Context Correction](#context-correction)
   - [Spelling Correction](#spelling-correction)
   - [Synonym Similarity](#synonym-similarity)
   - [Bigram Similarity](#bigram-similarity)
   - [Cosine Similarity](#cosine-similarity)
   - [Overall Sentence Similarity](#overall-sentence-similarity)
   - [Text Preprocessing](#text-preprocessing)
   - [Paragraph Similarity](#paragraph-similarity)
4. [Main Functions](#4-main-functions)

---

## **1. Imports**
The script uses the following imports:
```python
from gramformer import Gramformer
from transformers import pipeline
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
```
### Libraries Explained:
- **Gramformer**: Used for grammatical correction.
- **Transformers**: Utilized for spelling correction using pre-trained models.
- **SpaCy**: Helps with text processing and synonym generation.
- **Scikit-learn**: For computing cosine similarity.
- **Sentence Transformers**: For generating embeddings of sentences.

---

## **2. Modules and Libraries**
1. **Gramformer**:
   - Used to correct grammar in sentences.
   - Initialized with `models=1` (grammar correction model) and `use_gpu=False`.

2. **Transformers - Text2Text Generation**:
   - Provides spelling correction using the model `"oliverguhr/spelling-correction-english-base"`.

3. **Spacy NLP**:
   - Helps in text processing tasks like tokenization, lemmatization, and synonym generation.
   - Model used: `"en_core_web_sm"`.

4. **Sentence Transformers**:
   - Generates embeddings for sentences to compute cosine similarity.
   - Model: `'all-MiniLM-L6-v2'`.

---

## **3. Functions**

### **Context Correction (Long answers)**
#### Function: `correct_grammar(text)`
Corrects grammar using the Gramformer model.

**Parameters**:
- `text` (str): The input text to be corrected.

**Returns**:
- `str`: The grammatically corrected version of the input text.

---

### **Spelling Correction (Short answers)**
#### Function: `correct_spelling(text)`
Corrects spelling mistakes in short sentences using a pre-trained model.

**Parameters**:
- `text` (str): The input text to be corrected.

**Returns**:
- `str`: Text with spelling errors corrected.

---

### **Synonym Similarity**
#### Functions:
1. **`generate_synonyms(word)`**
   - Generates synonyms for a word based on similarity scores.
   - Threshold for similarity: `0.7`.

2. **`calculate_synonym_similarity(true_answer, student_answer)`**
   - Computes similarity between tokens of `true_answer` and `student_answer`.
   - Matches based on synonyms.

**Parameters**:
- `true_answer` (str): Reference text.
- `student_answer` (str): Text provided by the student.

**Returns**:
- `float`: Similarity score.

---

### **Bigram Similarity**
#### Functions:
1. **`generate_bigrams(word_list)`**
   - Generates bigrams (pairs of consecutive words) for a list of words.

2. **`bigram_similarity(text1, text2)`**
   - Computes similarity between bigrams of two texts.

**Parameters**:
- `text1` (str): First input text.
- `text2` (str): Second input text.

**Returns**:
- `float`: Bigram similarity score.

---

### **Cosine Similarity**
#### Function: `cos_similarity(sentence1, sentence2)`
Calculates the cosine similarity between embeddings of two sentences.

**Parameters**:
- `sentence1` (str): First sentence.
- `sentence2` (str): Second sentence.

**Returns**:
- `float`: Cosine similarity score (0 to 1).

---

### **Overall Sentence Similarity**
#### Function: `avg_similarity(text1, text2)`
Combines bigram, synonym, and cosine similarity to compute an overall similarity score.

**Parameters**:
- `text1` (str): First text.
- `text2` (str): Second text.

**Returns**:
- `float`: Weighted average similarity score.

---

### **Text Preprocessing**
#### Functions:
1. **`remove_stopwords(text)`**
   - Removes stopwords using SpaCy's default stopword list.
2. **`lemmatize_text(text)`**
   - Converts words to their base form (lemma).
3. **`preprocess_text(text)`**
   - Combines lemmatization and stopword removal for preprocessing.

**Parameters**:
- `text` (str): Input text.

**Returns**:
- `str`: Preprocessed text.

---

### **Paragraph Similarity**
#### Functions:
1. **`cluster_sentences(sentences, n)`**
   - Clusters sentences into `n` groups based on similarity.
   
2. **`compare_answers(student_answer, answer_key, max_marks)`**
   - Compares student answers with the answer key to calculate marks and confidence.

**Parameters**:
- `student_answer` (str): Student's answer.
- `answer_key` (str): Reference answer.
- `max_marks` (int): Maximum marks for the answer.

**Returns**:
- `int`: Total marks scored.
- `list[float]`: Confidence scores for each cluster.

---

## **4. Main Functions**
### **Evaluate Single Sentence**
#### Function: `evaluate_sentence(student_answer, answer_key)`
Evaluates a single sentence answer.

**Parameters**:
- `student_answer` (str): Student's answer.
- `answer_key` (str): Reference answer.

**Returns**:
- `int`: Marks scored.
- `float`: Confidence score.

---

### **Evaluate Paragraph**
#### Function: `evaluate_paragraph(student_answer, answer_key, max_marks)`
Evaluates a paragraph answer.

**Parameters**:
- `student_answer` (str): Student's answer.
- `answer_key` (str): Reference answer.
- `max_marks` (int): Maximum marks for the answer.

**Returns**:
- `int`: Marks scored.
- `list[float]`: Confidence scores.

---

### **Usage Example**
```python
student_ans = "The cat is sitting on the mat."
answer_key = "A cat is on a mat."

# Evaluate single sentence
marks, confidence = evaluate_sentence(student_ans, answer_key)
print(f"Marks: {marks}, Confidence: {confidence}")

# Evaluate paragraph
marks, confidences = evaluate_paragraph(student_ans, answer_key, max_marks=5)
print(f"Marks: {marks}, Confidence Scores: {confidences}")
``` 

---

This modular design ensures scalability for other NLP tasks like text classification, essay grading, or custom similarity computation.