print("\n\n>Trainer starting.");

import os 
import sys
import json
import InpResLSA 
import InpJobSBERT 

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import lightgbm as lgb
import numpy as np


resumes = []; #array filename of PDF
job_roles = []; #array String 
job_descriptions = []; #array String 
job_educations = []; #array String
relevance_labels = None; #2D array of relevance labels per resume

MODEL_PATH = "HyScreen_LSA_SBERT_v2.model"

# Read JSON input from stdin to get Input Resume and Jobs =============================
data = sys.stdin.read().strip()
if data:
	try:
		payload = json.loads(data);

		resumes = payload.get("resumes", []);
		job_roles = payload.get("job_roles", []);        
		job_descriptions = payload.get("job_descriptions", []);        
		relevance_labels = payload.get("relevance_labels", []);  
		job_educations = payload.get("job_educations", []);  

		sys.stdout.flush();

	except json.JSONDecodeError:
		print(json.dumps({"error": "Invalid JSON"}));
		sys.stdout.flush();
		sys.exit(1);


#relevance_labels = np.random.randint(0, 3, size=(len(resumes), len(jobs)));
relevance_labels = np.array(relevance_labels, ndmin=1) 
relevance_labels = relevance_labels.reshape((len(resumes), len(job_descriptions))) #To make it from array of [594] to 2d array [33, 18]


## Run LSA and SBERT for Resumes and Jobs =====================================
InpResLSA.run(resumes);
#InpJobSBERT.run(job_descriptions);

# SBERT embeddings for job descriptions
InpJobSBERT.run(job_descriptions);
desc_embeddings = InpJobSBERT.embeddings.copy();

# SBERT embeddings for job roles
InpJobSBERT.run(job_roles);
role_embeddings = InpJobSBERT.embeddings.copy();

# SBERT embeddings for job educations
InpJobSBERT.run(job_educations);
educ_embeddings = InpJobSBERT.embeddings.copy();

print("\n\n>Total Ranking Imports done.");


print("\n>Starting Matching and Ranking module. =======================================");
"""
	Job Description → LSA similarity → feature 1
	Job Description → SBERT similarity → feature 2
	> Combine (average / weights / ML ranker)
	> Combined Data to LightGBM Trainer
	> Output model
"""


print("Relevance labels shape:", relevance_labels.shape)
print("Resumes:", len(resumes))
print("Jobs Descriptions:", len(job_descriptions))

### Cosine Similarity with LSA + SBERT =========================
X = [];  # feature vectors
y = [];  # relevance labels
qids = [];  # query ids (one per job description)

for i, job_desc in enumerate(job_descriptions):
	# LSA score vector for resumes
	job_vec_lsa = InpResLSA.vectorizer.transform([job_desc]);
	job_vec_lsa = InpResLSA.svd.transform(job_vec_lsa);
	lsa_scores = cosine_similarity(InpResLSA.lsa_matrix, job_vec_lsa).flatten();

	# SBERT score vector for resumes
	#job_vec_sbert = InpJobSBERT.embeddings[i].reshape(1, -1);
	#sbert_scores = np.tile(InpJobSBERT.embeddings[i, 0], len(resumes)) 

	sbert_role_score = np.tile(role_embeddings[i, 0], len(resumes));	
	sbert_desc_score = np.tile(desc_embeddings[i, 0], len(resumes));
	sbert_educ_score = np.tile(educ_embeddings[i, 0], len(resumes));

	# combine features per resume
	for j in range(len(resumes)):
		#X.append([lsa_scores[j], sbert_scores[j]]);
		X.append([ lsa_scores[j], sbert_role_score[j], sbert_desc_score[j], sbert_educ_score[j] ]);
		y.append(relevance_labels[j][i]);
		qids.append(i);  # job i is the "query group"

X = np.array(X);
y = np.array(y);
qids = np.array(qids);

print("\nTraining data built:");
print("Features:", X.shape);
print("Labels:", y.shape);
print("Groups:", np.unique(qids));

# Prepare LightGBM dataset
train_data = lgb.Dataset(X, label=y, group=[np.sum(qids == i) for i in np.unique(qids)]);

# Train LambdaMART model
params = {
	"objective": "lambdarank",
	"metric": "ndcg",
	"boosting": "gbdt",
	"learning_rate": 0.05,
	"min_data_in_leaf": 1,
	"min_data_in_bin": 1,
	'monotone_constraints': (0, 0, 0, 1),
};
ranker = lgb.train(params, train_data, num_boost_round=100);

## Save model
ranker.save_model(MODEL_PATH);
print(f"\n\n=============================\n\n>Model saved to {MODEL_PATH}\n\n=============================\n");



print("\nresumes");
print(resumes);

print("\njob_roles");
print(job_roles);

print("\njob_descriptions");
print(job_descriptions);

print("\njob_educations");
print(job_educations);

print("\nrelevance_labels");
print(relevance_labels);

