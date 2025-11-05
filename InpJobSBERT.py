from sentence_transformers import SentenceTransformer

embeddings = None;

def run(sentences):
    global embeddings

    print("\n\n>Starting SBERT module. ===================================================");
    print("\n>SBERT Imports done.\n");

    # Load the model / Transformer Encoder
    #model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2');
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu');
    #GPU better.    

    # Convert sentences to embeddings / Token-Level Embedding and Pooling
    embeddings = model.encode(sentences);

    # Output the embeddings
    print(embeddings);	

    #Main Outputs
    sentences = sentences;
    embeddings = embeddings;
