print("\n\n>Classifier starting.");

import os 
import sys
import json
import InpResLSA 
import InpJobSBERT 

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import lightgbm as lgb
import numpy as np

import sqlite3
import time


resumes = []; #array filename of PDF
job_roles = []; #array String 
job_descriptions = []; #array String 
job_educations = []; #array String

MODEL_PATH = "HyScreen_LSA_SBERT_v3.model"

# Read JSON input from stdin to get Input Resume and Jobs =============================
data = sys.stdin.read().strip()
if data:
	try:
		payload = json.loads(data);

		resumes = payload.get("resumes", []);
		job_roles = payload.get("job_roles", []);        
		job_descriptions = payload.get("job_descriptions", []);  
		job_educations = payload.get("job_educations", []);  

		sys.stdout.flush();

	except json.JSONDecodeError:
		print(json.dumps({"error": "Invalid JSON"}));
		sys.stdout.flush();
		sys.exit(1);


# Display jobs:
print("Job Role and Description list:");
for i in range(len(job_roles)):
	print(f"{job_roles[i]} - {job_descriptions[i]} - {job_educations[i]}");

## Run LSA and SBERT for Resumes and Jobs =====================================
print("\n>LSA embeddings for resumes:");
InpResLSA.run(resumes, "");
#InpJobSBERT.run(job_descriptions);

# SBERT embeddings for job descriptions
print("\n>SBERT embeddings for job descriptions:");
InpJobSBERT.run(job_descriptions, "", "JOB_DESCRIPTONS");
desc_embeddings = InpJobSBERT.embeddings.copy();

# SBERT embeddings for job roles
print("\n>SBERT embeddings for job roles:");
InpJobSBERT.run(job_roles, "", "JOB_ROLES");
role_embeddings = InpJobSBERT.embeddings.copy();

# SBERT embeddings for job educations
print("\n>SBERT embeddings for job educations:");
InpJobSBERT.run(job_educations, "", "JOB_EDUCATION");
educ_embeddings = InpJobSBERT.embeddings.copy();

print("\n \n>Total Ranking Imports done.");


print("\n>Starting Matching and Ranking module. =======================================");
"""
	Job Description → LSA similarity → feature 1
	Job Description → SBERT similarity → feature 2
	> Combine (average / weights / ML ranker)
	> Combined Data to LightGBM >Trained Model<
	> Output results
"""


### Cosine Similarity with LSA + SBERT ========================================================

## Load model
ranker = lgb.Booster(model_file=MODEL_PATH)
print(f"Loaded model from {MODEL_PATH}")

X = [];  # feature vectors
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
		X.append([ lsa_scores[j], sbert_role_score[j], sbert_desc_score[j], sbert_educ_score[j] ]);
		qids.append(i);  # job i is the "query group"

X = np.array(X);
qids = np.array(qids);

### LambdaMart Relevance Ranking ===========================

# Predict relevance scores
preds = ranker.predict(X);

print("\n\n> Final Ranking ======================================================");

# Show ranked resumes per job description
for i in range(len(job_descriptions)):
	print(f"\n- Ranking for job: {job_roles[i]} - {job_descriptions[i]} - {job_educations[i]}");

	# get predictions for resumes of this job
	mask = qids == i;
	scores = preds[mask];
	
	# rank resumes by score
	ranked = sorted(zip(resumes, scores), key=lambda x: x[1], reverse=True)
#	for idx, (res, score) in enumerate(ranked, start=1):
##		print(f"#{idx} - {res}\t\t\t-\t {score:.4f}")
#		print(f"#{idx} -> {res[:25]} -> {score:.4f}");

	for idx, (resume, score) in enumerate(ranked, start=1):
		print(f"{resume} -> {score:.4f}");
		#print(f"#{idx} - {resume:<40}  -  {score:>8.4f}"); #full path
		#print(f"#{idx} - {os.path.basename(resume):<40}  -  {score:>8.4f}")




#Saving Output =========================================
print("\n\n> Outputing to database. ");

batch_no = int(time.time());
conn = sqlite3.connect("Result.db");
cursor = conn.cursor();

# Enable foreign key checks
cursor.execute("PRAGMA foreign_keys = ON;");

#  Insert batch number
print(f"\nInserting batch no: {batch_no}");
try:
	cursor.execute("INSERT INTO Batch (BATCH_NO) VALUES (?)", (batch_no,));	
except sqlite3.Error as e:
	print(f"Failed to insert resume {batch_no}: {e}");

#  Insert resumes 
print("\nInserting resumes:");
for resume in resumes:
	try:
		cursor.execute("INSERT INTO Candidate (RESUME, BATCH_NO) VALUES (?, ?)", (resume, batch_no));
		print(f"Inserted resume: {resume}");
	except sqlite3.Error as e:
		print(f"Failed to insert resume {resume}: {e}");

#  Insert jobs 
print("\nInserting jobs:");
for role, desc, educ in zip(job_roles, job_descriptions, job_educations):
    try:
        cursor.execute("INSERT OR IGNORE INTO Job_Role (JOB_ROLE, JOB_EDUCATION, BATCH_NO) VALUES (?, ?, ?)" , (role, educ, batch_no));
        cursor.execute("INSERT INTO Job_Description (JOB_ROLE, JOB_DESCRIPTION, BATCH_NO) VALUES (?, ?, ?)", (role, desc, batch_no));
        print(f"Inserted: Role='{role}' | Description='{desc}' | Education='{educ}'")
    except sqlite3.Error as e:
        print(f"Failed to insert Role='{role}': {e}");

print("\nInserting rankings:");
for i in range(len(job_descriptions)):
    mask = qids == i;
    scores = preds[mask];
    role = job_roles[i];
    desc = job_descriptions[i];

    for resume, score in zip(resumes, scores):
        try:
            #cursor.execute("INSERT OR REPLACE INTO Score (BATCH_NO, JOB_ROLE, JOB_DESCRIPTION, RESUME, SCORE) VALUES (?, ?, ?, ?, ?);",
            #               (batch_no, role, desc, resume, float(score)));
            cursor.execute("INSERT OR REPLACE INTO Score (RESUME, BATCH_NO, JOB_ROLE, JOB_DESCRIPTION, SCORE) VALUES (?, ?, ?, ?, ?);",
                            (resume, batch_no, role, desc, float(score)));
            print(f"Inserted ranking: Resume='{resume}' | Role='{role}' | Score={score:.4f}");
        except sqlite3.Error as e:
            print(f"Failed to insert ranking Resume='{resume}' Role='{role} Score={score:.4f}': {e}");

conn.commit();
conn.close();



#Percentages =========================

# Define your feature names (same order used during training)
feature_names = ['LSA_Similarity', 'SBERT_Role', 'SBERT_Description', 'SBERT_Education'];

# Get feature importances
importances = ranker.feature_importance(importance_type='gain');

# Convert to percentages
total = np.sum(importances);
percentages = (importances / total) * 100;

print("\n>Feature Importance (from saved model):");
for name, imp, pct in zip(feature_names, importances, percentages):
	print(f"{name}: {imp:.4f} gain ({pct:.2f}%)");
