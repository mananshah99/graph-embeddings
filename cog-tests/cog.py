# cog.py

import os
import sys
sys.path.insert(0, '../common')

import settings
import util

from tqdm import tqdm
import numpy as np

cogIDmap = {}
proteinIDmap = {}

def read_file(location, nlines=7881827):
    
    f = open(location, 'r')
    progress = tqdm(total=nlines)

    idx = 0
    for line in f:
        if idx == 0:
            progress.update(1)
            idx += 1
            continue
        else:
            line = line.split('\t')
            proteinID = line[0]
            cogID = line[3]

            if cogID in cogIDmap:
                cogIDmap[cogID].append(proteinID)
            else:
                cogIDmap[cogID] = [proteinID]

            if proteinID in proteinIDmap:
                proteinIDmap[proteinID].append(cogID)
            else:
                proteinIDmap[proteinID] = [cogID]

            idx += 1
            progress.update(1)
            progress.set_description('Reading COG ' + cogID)
    
    progress.close()
    f.close()

def n_cogs(pid1, pid2):
    try:
        protein_1 = proteinIDmap[pid1]
        protein_2 = proteinIDmap[pid2]
        return len(set(protein_1).intersection(set(protein_2))), 0
    except Exception as e:
        #print sys.exc_info()[0]
        return 0, 1

def _create_local_mapping(location, outer_loop=True):
    f = open(location, 'r')

    if outer_loop:
        nlines = util.count_lines(location)
        progress = tqdm(total=nlines)
        progress.set_description('Reading ' + location.split('/')[-1])

    ids = set()

    for line in f:
        line = line.split(' ')
        pid1 = line[0]
        pid2 = line[1]

        ids.add(pid1)
        ids.add(pid2)

        if outer_loop:
            progress.update(1)

    if outer_loop:
        progress.close()

    nids = len(ids)
    ids_list = list(ids)

    f.close()

    return ids_list

def intra_species_matrix(location):

    # generate a matrix
    # for every pair check if they share a cog
    # need a function to determine whether both are in a cog
   
    nke = 0
    ntot = 0

    mapping = _create_local_mapping(location)
    mat = np.ndarray(shape=(len(mapping), len(mapping)))
    progress = tqdm(total=(len(mapping)))
    progress.set_description('Creating matrix, %KE = ' + str(nke))
    
    for i in range(0, len(mapping)):
        for j in range(i, len(mapping)):
            pid1 = mapping[i]
            pid2 = mapping[j]
            nc, ke = n_cogs(pid1, pid2)
            
            nke += ke
            ntot += 1

            mat[i][j] = nc
            mat[j][i] = nc

        pke = 100 * float(nke) / float(ntot)
        spke = "{0:.5f}".format(pke)
        progress.set_description('Creating matrix, %KE = ' + spke)
        
        progress.update(1)
        
    progress.close()

    np.save(open('/dfs/scratch0/manans/10141_2.npz', 'wb+'), mat)

def cog_coappearance(pids1, pids2):
    cogs1 = set()
    cogs2 = set()
    for pid in pids1:
        try:
            cogs = proteinIDmap[pid]
            cogs1 |= set(cogs)
        except Exception as e:
            pass

    for pid in pids2:
        try:
            cogs = proteinIDmap[pid]
            cogs2 |= set(cogs)
        except:
            pass

    return len(cogs1.intersection(cogs2))

def inter_species_matrix(locations):

    # creates a matrix between N species 
    # look at all genes within each species -- use create_local_mapping

    loc_ids = []
    loc_mappings = []
    progress = tqdm(total=len(locations))

    for loc in locations:
        progress.set_description('Reading ' + loc.split('/')[-1])
        mapping = _create_local_mapping(loc, outer_loop=False) # mapping stores all the IDs 
        loc_mappings.append(mapping)
        loc_ids.append(loc.split('/')[-1].strip('.txt'))
        progress.update(1)

    progress.close()

    mat = np.ndarray(shape=(len(loc_mappings), len(loc_mappings)))

    progress = tqdm(total = len(loc_mappings))

    for i in range(0, len(loc_mappings)):
        for j in range(i, len(loc_mappings)):
            #print loc_ids[i], loc_ids[j], cog_coappearance(loc_mappings[i], loc_mappings[j])
            coo = cog_coappearance(loc_mappings[i], loc_mappings[j])
            mat[i][j] = coo
            mat[j][i] = coo
        
        progress.update(1)
        progress.set_description('Writing ' + loc_ids[i])

    progress.close()
    np.save(open('/dfs/scratch0/manans/inter_species_' + str(len(locations)) + '.npz', 'wb+'), mat)
    np.save(open('/dfs/scratch0/manans/inter_species_' + str(len(locations)) + '_labels.npz', 'wb+'), np.array(loc_ids))

def cogs_per_species(locations):

    loc_ids = []
    loc_mappings = []
    progress = tqdm(total = len(locations))

    for loc in locations:
        progress.set_description('Reading ' + loc.split('/')[-1])
        mapping = _create_local_mapping(loc, outer_loop = False)
        loc_mappings.append(mapping)
        loc_ids.append(loc.split('/')[-1].strip('.txt'))
        progress.update(1)

    progress.close()

    num_ortho_proteins = []
    num_total_proteins = []

    progress = tqdm(total = len(locations))

    for mapping in loc_mappings:
        # each mapping is a set of protein ids (pids)
        proteins = set()
        for pid in mapping:
            try:
                cogs = proteinIDmap[pid]

                # this protein is part of an orthologous group
                if len(cogs) > 0:
                    proteins.add(pid)
            except:
                pass

        num_ortho_proteins.append(len(proteins))
        num_total_proteins.append(len(mapping))

    progress.close()

    f = open('/dfs/scratch0/manans/cog_table.txt', 'wb+')

    for i in tqdm(range(0, len(loc_ids))):
        f.write(loc_ids[i] + "\t" + str(num_ortho_proteins[i]) + "\t" + str(num_total_proteins[i]) + "\t" + str(float(num_ortho_proteins[i]) / num_total_proteins[i]) + "\n")

    f.close()

def min_complete_cogs(locations):

    loc_ids = []      # ALL SPECIES IDs
    loc_mappings = [] # PROTEINS
    loc_cogs = []     # COGS

    cog_species_map = {}

    locations = locations[0:500]
    progress = tqdm(total = len(locations))

    for loc in locations:
        progress.set_description('Reading ' + loc.split('/')[-1])
        mapping = _create_local_mapping(loc, outer_loop = False)

        cogs = set()
        for pid in mapping:
            try:
                cid = proteinIDmap[pid]
                cogs |= set (cid)
            except:
                pass
 
        loc_mappings.append(mapping)
        loc_ids.append(loc.split('/')[-1].strip('.txt'))
        loc_cogs.append(cogs)

        species_id = loc.split('/')[-1].strip('.txt')
        for cog in cogs:
            if cog not in cog_species_map:
                cog_species_map[cog] = {species_id}
            else:
                cog_species_map[cog] |= {species_id}

        progress.update(1)

    progress.close()

    # greedy descent to find the COGs that eliminate the most species
    while len(loc_ids) > 0:
        all_values = cog_species_map.values()
        all_keys   = cog_species_map.keys()

        indices = range(len(all_values))
        sorted_vals = sorted(indices, key = lambda x: len(all_values[x]), reverse=True) 
        # index [0] is the index of the longest thing in the actual array
        
        idx = sorted_vals[0] # [-1] for the last element, [0] for the first element, etc. 
        key = list(all_keys)[idx]
        val = list(all_values)[idx]
        
        
        print "Removing COG", key
        print "\t Will impact ", len(val)

        # remove all of the species that contain this COG
        for v in val:
            try:
                loc_ids.remove(str(v))
            except:
                pass
       
       # these species have been accounted for by the identified COG,
       # so remove them from the other COG lists so we always pick the best next one
        vset = set(val)
        for k, v in cog_species_map.items():
            cog_species_map[k] -= vset
            # if empty
            if cog_species_map[k] == set():
                cog_species_map.pop(k, None)

        # remove the chosen COG
        cog_species_map.pop(key, None)


# 1 -- read the COG mapping file and PPI interaction networks (2,031 species)
read_file(settings.COG_MAPPING)
files = [(settings.INDIVIDUAL_PPI_DIR + '/' +  f) for f in os.listdir(settings.INDIVIDUAL_PPI_DIR) if os.path.isfile(os.path.join(settings.INDIVIDUAL_PPI_DIR, f))]

# 3 -- generate inter species matrix 
# inter_species_matrix(files)

# 4 -- generate intra species matrix (genefor any particular species)
# intra_species_matrix(files[0])

# 5 -- generate table of cog files 
# cogs_per_species(files)

# 6 -- question 1: largest number of species that participate in one COG
# print max([len(x) for x in cogIDmap.values()])  #=> 36429

# 7 -- question 2: minimum number of COGs to cover all species

min_complete_cogs(files)
