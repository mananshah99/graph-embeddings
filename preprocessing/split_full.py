import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/afs/cs.stanford.edu/u/manans/Desktop/graph-embeddings/common/')
import settings

protein_links = open(settings.PROTEIN_LINKS, 'r')
column_indices = [0, 1, 2, 4, 5, 7, 9, 11]
output_dir = settings.INDIVIDUAL_PPI_DIR

curr_n = 394
curr_i = 0
curr_f = open(output_dir + '/' + str(curr_n) + '.txt', 'wb')

progress = tqdm(total=1847117371)
for line in protein_links:
    progress.set_description('Processing %s' % str(curr_n))

    v = np.array(line.strip('\n').split(' '))
    v = v[column_indices]

    # don't include the header line
    if curr_i > 0:
        species_id = int(v[0].split('.')[0])

        # if we have a new ppi network
        if species_id != curr_n:
            curr_f.close()
            curr_n = species_id
            curr_f = open(output_dir + '/' + str(curr_n) + '.txt', 'wb')
    
    # write to the file
    if curr_i > 0:
        s = " ".join(v)
        s += '\n'
        curr_f.write(s)

    curr_i += 1
    progress.update(1)

