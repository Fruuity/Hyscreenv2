import os
import pymupdf
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize); #to remove truncation of data

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

def run(resumes, mode=None):
	global lsa_matrix, vectorizer, svd;

	foldername = "log"+mode;
	os.makedirs(foldername, exist_ok=True)

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

		#Resultfoldername = "outputlogs";
		corpus.append(lemmatized_words);


	###  TF-IDF (Term Frequency-Inverse Document Frequency) ==================================

	#tfidf = TfidfVectorizer();
	tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x);
	result = tfidf.fit_transform(corpus);

	#IDF Valuesfoldername = "outputlogs";
	"""
	print('\nIDF values:');
	for ele1, ele2 in zip(tfidf.get_feature_names_out(), tfidf.idf_):
		print(ele1, ':', ele2);
	"""

	#TF-IDF Values
	"""
	print('\nWord indexes:')
	print(tfidf.vocabulary_)
	print('\nTF-IDF value:')foldername = "outputlogs";
	print(result)
	"""

	#Formatting for display
	coordinateFormat = result.tocoo();	
	pd.set_option("display.max_rows", None);
	pd.set_option("display.max_columns", None)
	pd.set_option("display.width", None)              # disable wrapping
	pd.set_option("display.max_colwidth", None)       # do not truncate individual columns
	pd.set_option("display.expand_frame_repr", False) # avoid multi-line wrapping by columns
	df_TFIDF = pd.DataFrame({
	#	"Doc Index": coo.row,
		"Doc Name": [os.path.basename(resumes[i]) for i in coordinateFormat.row],
		"Word": [tfidf.get_feature_names_out()[col] for col in coordinateFormat.col],
		"Word Index": coordinateFormat.col,
		"TF-IDF value": coordinateFormat.data
	});
	
	#Display
	#print("\n",df_TFIDF);
	print(df_TFIDF.to_string(index=False))
	
	# Save TF-IDF COOrdinateFormat table
	df_TFIDF.to_csv(os.path.join(foldername, mode+"_LSA_TF-IDF.csv"), index=False)	


	### Singular Value Decomposition (SVD)
	n_components = 100 if result.shape[1] > 100 else result.shape[1] - 1;
	svd = TruncatedSVD(n_components=n_components, random_state=42);
	lsa_matrix = svd.fit_transform(result);


	# lsa_matrix is now (num_docs x n_components)
	print("\nLSA representation:");
	print(lsa_matrix);
	

	# save LSA Matrix
	df_LSA = pd.DataFrame(lsa_matrix);
	df_LSA.insert(0, "Doc Name", [os.path.basename(x) for x in resumes]);
	df_LSA.to_csv(os.path.join(foldername, mode+"_LSA_SVD-FEATURES.csv"), index=False, header=False);
	

	#print("\nExplained variance ratio (sum):", svd.explained_variance_ratio_.sum())
	
	
	#print("\nTopic List:");
	"""
	#Display Topics
	terms = tfidf.get_feature_names_out();	
	for i, comp in enumerate(svd.components_):  
		terms_in_comp = zip(terms, comp);
		#sorted_terms = sorted(terms_in_comp, key=lambda x: x[1], reverse=True)[:10];
		sorted_terms = sorted(terms_in_comp, key=lambda x: x[1], reverse=True);
		print("Resume {}: {}".format(os.path.basename(resumes[i]), " ".join([t for t, val in sorted_terms])));		
		#print("Resume {} ({}): {}".format(i + 1, os.path.basename(str(resumes[i])), " ".join([t for t, val in sorted_terms])))
		#print("Resume {} ({}): {}".format(i + 1, os.path.basename(os.fspath(resumes[i])).encode('utf-8', 'ignore').decode('utf-8'), " ".join([t for t, val in sorted_terms])));
		#print("Resume {} ({}): {}".format(i + 1, os.path.basename(resumes[i]).encode('ascii', 'ignore').decode(), " ".join([t for t, val in sorted_terms])))		
	pd.DataFrame(topic_rows).to_csv(os.path.join(foldername, mode+"_LSA_TOPICS.csv"), index=False, header=False)
	"""

	#Display Topics
	terms = tfidf.get_feature_names_out();		
	topic_rows = [];
	for i, doc_vec in enumerate(lsa_matrix):
		doc_term_weights = doc_vec @ svd.components_;  # shape = (num_terms,) #idk how this works
		#sorted
		#sorted_terms = [terms[j] for j in np.argsort(-doc_term_weights)]; # Sort terms by weight descending for this document #sorted for display only
		#print(f"{os.path.basename(resumes[i])}: {' '.join(sorted_terms)}");		
		#topic_rows.append([os.path.basename(resumes[i])] + sorted_terms);
		
		#unsorted
		#print(f"{os.path.basename(resumes[i])}: {' '.join(list(terms))}");		
		#topic_rows.append([os.path.basename(resumes[i])] + list(terms));

		#Sorted
		sorted_idx = np.argsort(-doc_term_weights); # sort terms by weight descending for this document #sorted for display only
		sorted_terms = [terms[j] for j in sorted_idx if doc_term_weights[j] != 0]; # filter zero-weight terms only

		if len(sorted_terms) == 0:
			sorted_terms = [terms[j] for j in sorted_idx]; # fallback to original just in case

		#print(f"{os.path.basename(resumes[i])}: {' '.join(sorted_terms)}");		
		topic_rows.append([os.path.basename(resumes[i])] + sorted_terms);

	
	pd.DataFrame(topic_rows).to_csv(os.path.join(foldername, mode+"_LSA_SORTED_TOPICS.csv"), index=False, header=False)

	# Main outputs
	lsa_matrix = lsa_matrix;
	vectorizer = tfidf;
	svd = svd;



