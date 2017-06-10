'''
processes node2vec embeddings

input: embedding file [# embeddings, # dimensions]
steps:
    -- for all files -- 
    1. separate each vector (length = # dimensions)
    2. add the vector to a general KMeans clustering framework
    -- for each file --
    3. create a histogram of length D = number of dimensions in KMeans
    4. increment frequencies in histogram corresponding to labels of each vector
    5. save histogram as representation of the graph
'''
