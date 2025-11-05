import os
import pymupdf
import numpy as np
from PIL import Image
import easyocr

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
nltk.download('punkt_tab');
nltk.download('stopwords');
nltk.download('wordnet');
nltk.download('averaged_perceptron_tagger_eng');

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from sklearn.decomposition import TruncatedSVD


lsa_matrix = vectorizer = svd = None;

def run(resumes):
	global lsa_matrix, vectorizer, svd;

	#### Latent Semantic Analysis (LSA) ==================================================

	corpus = []; #string container for all the processed words of all documents

	for resume in resumes:
		text = "";

		# Try using text parser
		doc = pymupdf.open(resume); 
		if(text==""):
			for page in doc: 	
				text += page.get_text(); 
		
		#If text parser doesn't work, use OCR
		if(text==""): 
			print(">Text parsing yields no result. Trying OCR.")
			#reader = easyocr.Reader(['en']);
			reader = easyocr.Reader(['en'], gpu=False);
			for pageNum in range(len(doc)):
				pix = doc[pageNum].get_pixmap();
				img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n);
				result = reader.readtext(img, detail = 0);
				for word in result:
					text+=word+" ";

		## Tokenization ================================================================
		text = text.lower();
		tokens = nltk.word_tokenize(text);

		## Stopword removal =============================================================
		stop_words = set(stopwords.words('english'));
		filtered_tokens  = [t for t in tokens if t.isalpha() and t not in stop_words];

		## Lemmatization ================================================================
		lemmatizer = WordNetLemmatizer();
		pos_tags = pos_tag(filtered_tokens);

		def get_wordnet_pos(tag):
			if tag.startswith('J'):
				return wordnet.ADJ;
			elif tag.startswith('V'):
				return wordnet.VERB;
			elif tag.startswith('N'):
				return wordnet.NOUN;
			elif tag.startswith('R'):
				return wordnet.ADV;
			else:
				return wordnet.NOUN;

		lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags];
		#print(lemmatized_words);

		#Result
		corpus.append(lemmatized_words);


	###  TF-IDF (Term Frequency-Inverse Document Frequency) ==================================

	#tfidf = TfidfVectorizer();
	tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x);
	result = tfidf.fit_transform(corpus);

	#IDF Values
	"""
	print('\nIDF values:');
	for ele1, ele2 in zip(tfidf.get_feature_names_out(), tfidf.idf_):
		print(ele1, ':', ele2);
	"""

	#TF-IDF Values
	"""
	print('\nWord indexes:')
	print(tfidf.vocabulary_)
	print('\nTF-IDF value:')
	print(result)
	"""

	#Formatting for display
	coo = result.tocoo();
	df = pd.DataFrame({
	#	"Doc Index": coo.row,
		"Doc Name": [os.path.basename(resumes[i]) for i in coo.row],
		"Word": [tfidf.get_feature_names_out()[col] for col in coo.col],
		"Word Index": coo.col,
		"TF-IDF value": coo.data
	});
	pd.set_option("display.max_rows", None);

	#Display
	print("\n",df);

	#df.to_csv("tfidf_output.csv", index=False)

	### Singular Value Decomposition (SVD)

	n_components = 100 if result.shape[1] > 100 else result.shape[1] - 1;
	svd = TruncatedSVD(n_components=n_components, random_state=42);
	lsa_matrix = svd.fit_transform(result);

	# lsa_matrix is now (num_docs x n_components)
	print("\nLSA representation:");
	print(lsa_matrix);

	#print("\nExplained variance ratio (sum):", svd.explained_variance_ratio_.sum())

	print("\nTopic List:");

	terms = tfidf.get_feature_names_out()
	for i, comp in enumerate(svd.components_[:10]):  # first 10 topics
		terms_in_comp = zip(terms, comp);
		sorted_terms = sorted(terms_in_comp, key=lambda x: x[1], reverse=True)[:10];
		print("Resume {}: {}".format(i+1, " ".join([t for t, val in sorted_terms])));

	# Main outputs
	lsa_matrix = lsa_matrix;
	vectorizer = tfidf;
	svd = svd;


