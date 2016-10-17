# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 10:01:43 2016

@author: vladimir
"""

import numpy as np
from scipy.sparse import lil_matrix
from matplotlib import pyplot as plt
import networkx as nx
from fim import apriori

def test_data():
    data = [ [ 1, 2, 3 ], [ 1, 4, 5 ], [ 2, 3, 4 ], [ 1, 2, 3, 4 ],[ 2, 3 ],[ 1, 2, 4 ],[ 4, 5 ],[ 1, 2, 3, 4 ],[ 3, 4, 5 ],[ 1, 2, 3 ] ]
    return data
   
###############################################################################
# Read data  
def read_data():
    result=[]
    with open('../Data/marketing.data','r') as f:
        for line in f:
            try:
                result.append(list(map(int,line.split(','))))
            except ValueError: pass
        return result
###############################################################################
#data = test_data()
data = read_data()
 
###############################################################################
#convert to vector space format.
#map 14 attributes to 50 variables
def convert(x):
    y = []
    if x[0] < 5: y.append(1)
    else : y.append(2)
    if x[1] == 1: y.append(3) 
    else: y.append(4)
    if x[2] : y.append(4 + x[2])
    if x[3] < 4: y.append(10)
    else : y.append(11)
    if x[4] <= 3: y.append(12)
    else : y.append(13)
    if x[5] : y.append(13 + x[5])
    if x[6] < 3: y.append(23)
    else : y.append(24)    
    if x[7] == 1: y.append(25)
    elif x[7] == 2: y.append(26)
    else : y.append(27)
    if x[8] < 5 : y.append(28)
    else : y.append(29)
    if x[9] <= 4 : y.append(30)
    else : y.append(31)
    if x[10] == 1: y.append(32)
    elif x[10] == 2: y.append(33)
    else : y.append(34)
    if x[11] : y.append(35 + x[11])
    if x[12] : y.append(39 + x[12])
    if x[13] == 1: y.append(48)
    elif x[13] == 2: y.append(49)
    else : y.append(50)
    return y

#newdata = []    
#print(data)
data = list(map(convert,data))
#print(data)
###############################################################################
# Some basic data analysis
###############################################################################
'''
# Find items list
items = np.unique([item for sublist in data for item in sublist])

# Size of data
N_baskets = len(data)
M_items = len(items)


###############################################################################
# Convert data into vector space format.  We will use a sparse boolean matrix
#   to represent data
###############################################################################
H =lil_matrix((N_baskets,M_items), dtype=np.bool)
for i in range(0,len(data)):
    for j in list(map(int,data[i])):
        H[i,j-1] = True

# Plot this matrix
plt.figure(1)
plt.subplot(121)
plt.spy(H)
plt.title('Vector representation')
plt.xlabel('Items')
plt.ylabel('Baskets')
plt.show()


###############################################################################
# Convert data into graph format
###############################################################################
g = nx.Graph()
a=['b_'+str(i) for i in range(N_baskets)]
b=['i_'+str(j) for j in range(M_items)]
g.add_nodes_from(a,bipartite=0)
g.add_nodes_from(b,bipartite=1)

i=0
for basket in data:
    for item in basket:
            g.add_edge(a[i], b[list(items).index(item)])
    i+=1

# Draw this graph
pos_a={}
x=0.100
const=0.100
y=1.0
for i in range(len(a)):
    pos_a[a[i]]=[x,y-i*const]

xb=0.500
pos_b={}
for i in range(len(b)):
    pos_b[b[i]]=[xb,y-i*const]

plt.subplot(121)
nx.draw_networkx_nodes(g,pos_a,nodelist=a,node_color='r',node_size=300,alpha=0.8)
nx.draw_networkx_nodes(g,pos_b,nodelist=b,node_color='b',node_size=300,alpha=0.8)

# edges
pos={}
pos.update(pos_a)
pos.update(pos_b)
nx.draw_networkx_edges(g,pos,edgelist=nx.edges(g),width=1,alpha=0.8,edge_color='g')
nx.draw_networkx_labels(g,pos,font_size=10,font_family='sans-serif')

plt.title('Graph representation')
plt.show()

'''
###############################################################################
# Now do rule finding
###############################################################################

#frequent_itemset = apriori(data, supp=10, zmin=2, target='s', report='a')
rules = apriori(data, supp=10, zmin=2, zmax=5, target='r', report='SCl')

#print(frequent_itemset)
#print(rules)

###############################################################################
#sort the result
r = sorted(rules,reverse=True,key=lambda x:x[4])
print(r)
print(len(r))