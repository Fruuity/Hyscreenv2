import os
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize); #to remove truncation of data
import csv

from sentence_transformers import SentenceTransformer

embeddings = None;


def run(sentences, mode = None, variant = None):
    global embeddings;

    foldername = "log"+mode;
    os.makedirs(foldername, exist_ok=True)

    print("\n>Starting SBERT module. ===================================================");    
    print("\n>SBERT Imports done.\n");

    # Load the model / Transformer Encoder
    #model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2');
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu');
    #GPU better.    

    # Convert sentences to embeddings / Token-Level Embedding and Pooling
    embeddings = model.encode(sentences);

    # Output the embeddings
    #print(embeddings);	
    a = 0;
    for embedding in embeddings:
        print([sentences[a]] + embedding.tolist());
        a+=1;

    if(mode!=None):
        with open( os.path.join(foldername, mode+'_SBERT_'+variant+'.csv'), 'w', newline='') as file:
            writer = csv.writer(file);     
            a = 0;
            for embedding in embeddings:
                writer.writerow([sentences[a]] + embedding.tolist());
                a += 1;
            

    #Main Outputs
    sentences = sentences;
    embeddings = embeddings;
