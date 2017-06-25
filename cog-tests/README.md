
# Clusters of Orthologous Genes (COGs) and their Properties

This notebook contains visualizations of analyses constructed from the main COG mapping file. All code used to generate these outputs (and commented functions) is located at ```/afs/cs.stanford.edu/u/manans/Desktop/graph-embeddings/cog-tests/cog.py```.


```python
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (12, 12)
plt.ion()
```

## Intra-Species Analysis

In order to determine how protiens are inter-related within a single species, a matrix was constructed for the sample species with NCBI ID 10141, in which each index consisted of the number of orthologous groups that the two proteins shared. More precisely, $$M_{ij} = M_{ji} = \mathbf{card}(\text{COG}_i \cap \text{COG}_j)$$ where $\text{COG}_x$ is the set of orthologous groups that contain protein $x$. Proteins not included in any orthologous group are represented by the null set $\emptyset$, and their intersection with themselves (along the diagonal) is counted as 0.


```python
intra_species_10141 = np.load('10141.npz')
plt.matshow(intra_species_10141, aspect='auto')
```




    <matplotlib.image.AxesImage at 0x7f318c39ec10>




![png](output_5_1.png)


## Inter-Species Analysis

In order to determine the relationships of different species based on the COGs that their proteins share, a matrix was constructed both for a small subset of 100 species (of dimension $100 \times 100$) and for the entire set of 2,031 species (of dimension $2031 \times 2031$). Specifically, we have $$M_{ij} = M_{ji} = \textbf{card} \left( \bigcup_{n=1}^{N_i}\text{COG}_{n} \cap \bigcup_{m=1}^{N_j}\text{COG}_{m} \right) $$ where $N_i$ and $N_j$ are the number of proteins in species $i$ and $j$, respectively. All diagonals are artificially set to 0 so that the variances in other (more instructive) interactions can be more readily observed.


```python
inter_species_100 = np.load('inter_species_100.npz')

for (i, j), z in np.ndenumerate(inter_species_100):
    if i == j:
        inter_species_100[i][j] = 0
plt.matshow(inter_species_100, aspect='auto')

# For numerical visualization
# for (i, j), z in np.ndenumerate(matrix1):
#    plt.text(j, i, str(int(z)), ha='center', va='center')
```




    <matplotlib.image.AxesImage at 0x7f318c147890>




![png](output_8_1.png)



```python
inter_species_2031 = np.load('inter_species_2031.npz')
for (i, j), z in np.ndenumerate(inter_species_2031):
    if i == j:
        inter_species_2031[i][j] = 0
plt.matshow(inter_species_2031, aspect='auto')
```




    <matplotlib.image.AxesImage at 0x7f318c073c50>




![png](output_9_1.png)


## Matrix Clustering and Prediction

Now that we've obtained a matrix of interactions between all 2,031 species, we can treat each row as a feature vector for its corresponding species (that is, species $i$ is represented by feature vector $M[i][:]$). These vectors are clustered via KMeans into three groups to provide representation for the three domains Bacteria, Archaea, and Eukaryota, and the clusters are evaluated for discriminatory potential. 


```python
from sklearn.cluster import KMeans

# 3 clusters to ideally separate archaea, bacteria, and eukarya
kmeans = KMeans(n_clusters=3, random_state=0).fit(inter_species_2031)
labels = kmeans.predict(inter_species_2031)

# row_dict represents the true IDs of the rows
row_dict = np.load('inter_species_2031_labels.npz')
```


```python
clusters = {}
n = 0
for item in labels:
    if item in clusters:
        clusters[item].append(row_dict[n])
    else:
        clusters[item] = [row_dict[n]]
    n +=1

print len(clusters[0]), len(clusters[1]), len(clusters[2])
```

    845 209 977


The NCBITaxa SQL database is used to obtain the lineage for any particular ID and identify the domain associated with the ID. 


```python
from ete2 import NCBITaxa
ncbi = NCBITaxa('/dfs/scratch0/manans/.etetoolkit/taxa.sqlite')

def id_to_domain(id_str):
    lineage = ncbi.get_lineage(id_str)
    try:
        domain_map = ncbi.get_taxid_translator([lineage[2]])
        return domain_map[lineage[2]]
    except:
        domain_map = ncbi.get_taxid_translator([lineage[0]])
        return domain_map[lineage[0]]
```


```python
cluster_mapping = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

for item in clusters:
    print "Cluster ", item
    for i in clusters[item]:
        domain = id_to_domain(str(i))
        if domain == 'Bacteria':
            cluster_mapping[item][0] += 1
        elif domain == 'Archaea':
            cluster_mapping[item][1] += 1
        elif domain == 'Eukaryota':
            cluster_mapping[item][2] += 1
```

    Cluster  0
    Cluster  1
    Cluster  2



```python
cm = np.array(cluster_mapping)
cm
```




    array([[698, 114,  29],
           [  0,   0, 206],
           [977,   0,   0]])



As seen above, the Eukaryota class seems to be significantly (and easily) differentiated from the Bacteria and Archaea classes, but the distinctions between Bacteria and Archaea are quite hard to identify. 

## Viability of Only Considering COG-Linked Proteins in Species

Finally, we considered the feasiblity of only analyzing COG-linked proteins (CLPs) when making predictions about species. For each species, protein $i$ was designated a CLP if $\textbf{card}(\text{COG}_i) > 0$. 


```python
import pandas as pd

cog_table = pd.read_csv('cog_table.txt', sep ='\t', header = 0, names = ["Species ID", "# CLPs", "# Proteins", "%CLPs"])
cog_table
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Species ID</th>
      <th># CLPs</th>
      <th># Proteins</th>
      <th>%CLPs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>272563</td>
      <td>3333</td>
      <td>3698</td>
      <td>0.901298</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10090</td>
      <td>19871</td>
      <td>20648</td>
      <td>0.962369</td>
    </tr>
    <tr>
      <th>2</th>
      <td>340099</td>
      <td>2106</td>
      <td>2230</td>
      <td>0.944395</td>
    </tr>
    <tr>
      <th>3</th>
      <td>316275</td>
      <td>3346</td>
      <td>3753</td>
      <td>0.891553</td>
    </tr>
    <tr>
      <th>4</th>
      <td>699184</td>
      <td>1454</td>
      <td>2126</td>
      <td>0.683913</td>
    </tr>
    <tr>
      <th>5</th>
      <td>573065</td>
      <td>2919</td>
      <td>3425</td>
      <td>0.852263</td>
    </tr>
    <tr>
      <th>6</th>
      <td>290512</td>
      <td>2097</td>
      <td>2249</td>
      <td>0.932414</td>
    </tr>
    <tr>
      <th>7</th>
      <td>582744</td>
      <td>2509</td>
      <td>2805</td>
      <td>0.894474</td>
    </tr>
    <tr>
      <th>8</th>
      <td>547558</td>
      <td>1899</td>
      <td>1973</td>
      <td>0.962494</td>
    </tr>
    <tr>
      <th>9</th>
      <td>314256</td>
      <td>3460</td>
      <td>3745</td>
      <td>0.923899</td>
    </tr>
    <tr>
      <th>10</th>
      <td>873517</td>
      <td>2064</td>
      <td>2311</td>
      <td>0.893120</td>
    </tr>
    <tr>
      <th>11</th>
      <td>420662</td>
      <td>3489</td>
      <td>3810</td>
      <td>0.915748</td>
    </tr>
    <tr>
      <th>12</th>
      <td>190304</td>
      <td>1775</td>
      <td>2047</td>
      <td>0.867123</td>
    </tr>
    <tr>
      <th>13</th>
      <td>684719</td>
      <td>1218</td>
      <td>1415</td>
      <td>0.860777</td>
    </tr>
    <tr>
      <th>14</th>
      <td>663321</td>
      <td>1457</td>
      <td>1675</td>
      <td>0.869851</td>
    </tr>
    <tr>
      <th>15</th>
      <td>945543</td>
      <td>3774</td>
      <td>4296</td>
      <td>0.878492</td>
    </tr>
    <tr>
      <th>16</th>
      <td>331636</td>
      <td>1488</td>
      <td>1938</td>
      <td>0.767802</td>
    </tr>
    <tr>
      <th>17</th>
      <td>667128</td>
      <td>1891</td>
      <td>2042</td>
      <td>0.926053</td>
    </tr>
    <tr>
      <th>18</th>
      <td>517417</td>
      <td>1946</td>
      <td>2026</td>
      <td>0.960513</td>
    </tr>
    <tr>
      <th>19</th>
      <td>390333</td>
      <td>1233</td>
      <td>1493</td>
      <td>0.825854</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1003195</td>
      <td>4044</td>
      <td>5681</td>
      <td>0.711847</td>
    </tr>
    <tr>
      <th>21</th>
      <td>266940</td>
      <td>3800</td>
      <td>4452</td>
      <td>0.853549</td>
    </tr>
    <tr>
      <th>22</th>
      <td>552398</td>
      <td>2767</td>
      <td>3141</td>
      <td>0.880930</td>
    </tr>
    <tr>
      <th>23</th>
      <td>880447</td>
      <td>666</td>
      <td>875</td>
      <td>0.761143</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1032480</td>
      <td>4529</td>
      <td>5286</td>
      <td>0.856792</td>
    </tr>
    <tr>
      <th>25</th>
      <td>7739</td>
      <td>33891</td>
      <td>34332</td>
      <td>0.987155</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1042156</td>
      <td>2267</td>
      <td>2594</td>
      <td>0.873940</td>
    </tr>
    <tr>
      <th>27</th>
      <td>656519</td>
      <td>2082</td>
      <td>2279</td>
      <td>0.913559</td>
    </tr>
    <tr>
      <th>28</th>
      <td>575594</td>
      <td>1394</td>
      <td>1627</td>
      <td>0.856792</td>
    </tr>
    <tr>
      <th>29</th>
      <td>261317</td>
      <td>355</td>
      <td>359</td>
      <td>0.988858</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>1005048</td>
      <td>4092</td>
      <td>4369</td>
      <td>0.936599</td>
    </tr>
    <tr>
      <th>2001</th>
      <td>416269</td>
      <td>1900</td>
      <td>2010</td>
      <td>0.945274</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>243274</td>
      <td>1707</td>
      <td>1850</td>
      <td>0.922703</td>
    </tr>
    <tr>
      <th>2003</th>
      <td>167555</td>
      <td>1369</td>
      <td>2145</td>
      <td>0.638228</td>
    </tr>
    <tr>
      <th>2004</th>
      <td>384616</td>
      <td>1914</td>
      <td>1960</td>
      <td>0.976531</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>1000588</td>
      <td>1615</td>
      <td>1988</td>
      <td>0.812374</td>
    </tr>
    <tr>
      <th>2006</th>
      <td>525898</td>
      <td>2092</td>
      <td>2253</td>
      <td>0.928540</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>273121</td>
      <td>1927</td>
      <td>2040</td>
      <td>0.944608</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>164757</td>
      <td>4706</td>
      <td>5714</td>
      <td>0.823591</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>169963</td>
      <td>2557</td>
      <td>2835</td>
      <td>0.901940</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>419947</td>
      <td>5726</td>
      <td>8277</td>
      <td>0.691797</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>212042</td>
      <td>728</td>
      <td>1130</td>
      <td>0.644248</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>148304</td>
      <td>4970</td>
      <td>5100</td>
      <td>0.974510</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>5786</td>
      <td>5662</td>
      <td>5785</td>
      <td>0.978738</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>633148</td>
      <td>1305</td>
      <td>1380</td>
      <td>0.945652</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>657314</td>
      <td>2595</td>
      <td>3108</td>
      <td>0.834942</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>858215</td>
      <td>2207</td>
      <td>2334</td>
      <td>0.945587</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>523850</td>
      <td>1907</td>
      <td>1967</td>
      <td>0.969497</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>315730</td>
      <td>3794</td>
      <td>5059</td>
      <td>0.749951</td>
    </tr>
    <tr>
      <th>2019</th>
      <td>745277</td>
      <td>4124</td>
      <td>4317</td>
      <td>0.955293</td>
    </tr>
    <tr>
      <th>2020</th>
      <td>585531</td>
      <td>2558</td>
      <td>3051</td>
      <td>0.838414</td>
    </tr>
    <tr>
      <th>2021</th>
      <td>589924</td>
      <td>2208</td>
      <td>2456</td>
      <td>0.899023</td>
    </tr>
    <tr>
      <th>2022</th>
      <td>653045</td>
      <td>6586</td>
      <td>8368</td>
      <td>0.787046</td>
    </tr>
    <tr>
      <th>2023</th>
      <td>650150</td>
      <td>1436</td>
      <td>1695</td>
      <td>0.847198</td>
    </tr>
    <tr>
      <th>2024</th>
      <td>526224</td>
      <td>2025</td>
      <td>2775</td>
      <td>0.729730</td>
    </tr>
    <tr>
      <th>2025</th>
      <td>333990</td>
      <td>2094</td>
      <td>2367</td>
      <td>0.884664</td>
    </tr>
    <tr>
      <th>2026</th>
      <td>653733</td>
      <td>2219</td>
      <td>2548</td>
      <td>0.870879</td>
    </tr>
    <tr>
      <th>2027</th>
      <td>457429</td>
      <td>5135</td>
      <td>6762</td>
      <td>0.759391</td>
    </tr>
    <tr>
      <th>2028</th>
      <td>546271</td>
      <td>2019</td>
      <td>2613</td>
      <td>0.772675</td>
    </tr>
    <tr>
      <th>2029</th>
      <td>5518</td>
      <td>7500</td>
      <td>7889</td>
      <td>0.950691</td>
    </tr>
  </tbody>
</table>
<p>2030 rows × 4 columns</p>
</div>




```python
print cog_table["# CLPs"].mean(), cog_table["# Proteins"].mean(), cog_table["%CLPs"].mean()
```

    3603.49852217 4065.18472906 0.862479417084


Based on this simple analysis, it seems that analyzing only CLPs would not provide a significant benefit over analyzing a random subset of proteins. 
