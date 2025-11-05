print("\n\n>Mission start.");

import os 
import csv
import json
import subprocess

import numpy as np

resumes = []; #array filename of PDF
job_roles = []; #array String 
job_descriptions = []; #array String 
job_educations = [];  #array String 
relevance_labels = [];

#Model file location
#MODEL_PATH = "HyScreen_LSA_SBERT.model"

#Get current file location
dir = os.path.dirname(os.path.abspath(__file__));

## Input resume ==============================================================
#Scan "Sample Data" folder
for file in os.listdir(dir+"/Resumes/"):
	if(file.lower().endswith(".pdf")):
		resumes.append(dir+"/Resumes/"+file);

#Display Resume
print("\n>Scanned Resumes:\nIndex#\tDocument Name");
for resume in resumes:	
	#print(resumes.index(resume),"\t",resume);
	print(resumes.index(resume),"\t",os.path.basename(resume));
print("\n");	


## Input Jobs and Relevance Labels ===========================================

#with open(dir+"/job_descriptions.csv", "r", encoding="utf-8") as f:
with open(dir+"/Training_Data.csv", "r", encoding="utf-8") as f:
	reader = csv.reader(f);	
	for row in reader:
		if len(row) > 1 and row[1].strip():
			role = row[1].strip();
			desc = row[2].strip();
			educ = row[6].strip();
			key = (role, desc, educ);
			if key not in list(zip(job_roles, job_descriptions, job_educations)):
				job_roles.append(role);
				job_descriptions.append(desc);
				job_educations.append(educ);
			
			#relevance_labels.append( ( float(row[3].strip()) + float(row[4].strip()) + float(row[5].strip()) ) / 3 );
			relevance_labels.append(
			    int(round( ((float(row[3].strip()) + float(row[4].strip()) + float(row[5].strip())) / 3) * 10) )
			)
			


# Display jobs:
print("\nJob Role and Description list:");
for i in range(len(job_roles)):
	print(f"{job_roles[i]} - {job_descriptions[i]} - {job_educations[i]}");
print("\n");


##TODO
# Create JSON payload for TotalRankingTrainer
payload = {
	"resumes": resumes,
	"job_roles": job_roles,	
	"job_descriptions": job_descriptions,
	"job_educations": job_educations,
	"relevance_labels": relevance_labels
}

### Training Model =========================================================

# Call the trainer and pass JSON through stdin
trainer_path = os.path.join(dir, "TotalRankingTrainer.py");

print("\n>Launching TotalRankingTrainer.py ...\n");

result = subprocess.run(
    ["python3", trainer_path],
    input=json.dumps(payload),
    text=True,
    capture_output=True
);
# Display trainer output
print(result.stdout);
if result.stderr:
    print("Errors:", result.stderr);


"""
### Using the Model =========================================================

print("\n>Launching TotalRankingClassifier.py ...\n");

# Call the classifier and pass JSON through stdin
classifier_path = os.path.join(dir, "TotalRankingClassifier.py");

result = subprocess.run(
    ["python3", classifier_path],
    input=json.dumps(payload),
    text=True,
    capture_output=True
);

# Display classifier output
print(result.stdout);
if result.stderr:
    print("Errors:", result.stderr)
"""
