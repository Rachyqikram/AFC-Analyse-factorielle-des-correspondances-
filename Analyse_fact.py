#!/usr/bin/env python
# coding: utf-8

# In[71]:


import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from fanalysis.ca import CA
import scipy


#  Nous chargeons la feuille« AFC» du classeur « AFC.xlsx ».

# In[72]:


Data = pandas.read_excel("AFC.xlsx",sheet_name="AFC",index_col=0)
print(Data);


# calculons les totaux en ligne et  les totaux en colonnes.

# In[73]:


#calcul des totaux en ligne
tot_lig = numpy.sum(Data.values,axis=1)
print(tot_lig)


#calcul des totaux en colonne
tot_col = numpy.sum(Data.values,axis=0)
print(tot_col)


# Analyse des profils lignes:
# 

# In[74]:


# calculons les proportions lignes
prof_lig = numpy.apply_along_axis(arr=D.values,axis=1,func1d=lambda x:x/numpy.sum(x))
print(prof_lig)


# représentation graphique  des profils des groupes d'âge

# In[75]:


somme = numpy.zeros(shape=(prof_lig.shape[0]))
for i in range(prof_lig.shape[1]):
 plt.barh(range(prof_lig.shape[0]),prof_lig[:,i],left=somme)
 somme = somme + prof_lig[:,i]

plt.yticks(range(prof_lig.shape[0]),D.index)
plt.show()


#  
# Dans notre exemple , on remarque que les profils des groupes d'âge 25-34, 35-44 et 45-54 sont proches les uns des autre

# calulons les profils marginals corresp. 

# In[76]:


prof_marg_lig = tot_col/numpy.sum(tot_col)
print(prof_marg_lig)


#  caculons la distance du KHI-2 (La  distance  entre  profils) entre la tranche 35-44(2) et 44-54(3).

# In[77]:


print(numpy.sum((prof_lig[2,:]-prof_lig[3,:])**2/prof_marg_lig))


# calculons la distance du KHI-2 entre la tranche(2) et 25-34(1).

# In[78]:


print(numpy.sum((prof_lig[2,:]-prof_lig[1,:])**2/prof_marg_lig))


# calculons la distance entre paires de modalités lignes.

# In[79]:


distPairesLig = numpy.zeros(shape=(prof_lig.shape[0],prof_lig.shape[0]))
#double boucle
for i in range(prof_lig.shape[0]-1):
  for j in range(0,prof_lig.shape[0]):
     distPairesLig[i,j] = numpy.sum((prof_lig[i,:]-prof_lig[j,:])**2/prof_marg_lig)
 #distPairesLig[j,i] = distPairesLig[i,j]
#affichage
print(pandas.DataFrame(distPairesLig,index=D.index,columns=D.index))


# affichage sous forme de heatmap.
# Une représentation graphique sous forme de « heatmap » donne une vision globale des écarts.

# In[80]:


sns.heatmap(distPairesLig,vmin=0,vmax=numpy.max(distPairesLig),linewidth=0.1,cmap='PiYG',xticklabels=D.index,
yticklabels=D.index)


# Les opinions de la tranche d'âge 25-34, 35-44 et 45-54  présentent des structures de choix assez proches . Il en est de même aussi, à un  moindre degré cependant , entre le  groupes  d'âge 16-24 

# Distance à l’origine

# In[81]:


distoLig = numpy.apply_along_axis(arr=prof_lig,axis=1,func1d=lambda x:numpy.sum((x-prof_marg_lig)**2/prof_marg_lig))
#affichage
print(pandas.DataFrame(distoLig,index=D.index))


# Calculons le poids et l'inertie des lignes.

# In[82]:


poidsLig = tot_lig/numpy.sum(tot_lig)

#inertie des lignes
inertieLig = distoLig * poidsLig
#affichage
print(pandas.DataFrame(numpy.transpose([distoLig,poidsLig,inertieLig]),columns=['Disto2','Poids','Inertie'] ,index=D.index))


#  Les poids sont des proportions marginales utilisées pour pondérer les profils des points lors du calcul des distances. Plus la distance à l'origine est grande, plus le profil de la catégorie est différent du profil moyen (plus la catégorie participe à la dépendance entre les deux variables). Les groupes d'âge 25-34, 35-44 et 45-54 ont la distance la plus courte à l'origine, ce qui indique que les profils de ces groupes sont proches du profil moyen.
# 

# In[83]:


#total inertie
tot_InertieLig = numpy.sum(inertieLig)
print(tot_InertieLig)


# Nous réalisons une AFC avec le package « fanalysis ».

# Affichons les valeurs propres.
# 
# Affichons la fraction d’inertie restituée par facteur.
# 
# Affichons la fraction d’inertie restituée cumulée.

# In[84]:


afc = CA(row_labels=D.index,col_labels=D.columns)
afc.fit(D.values)
#information restituée sur les facteurs
print(afc.eig_)




# In[85]:


#coordonnees des modalites lignes
print(pandas.DataFrame(afc.row_coord_,index=D.index))


# 
# Nous observons pour chaque ligne de la matrice « eig_ »: les valeurs propres des "min(7-1, 4-1)= 3" facteurs, la fraction
# 
# d’inertie restituée par facteur et cumulée. La variance du 1er facteur est égale à (𝜆1 = 0.09466), elle représente déjà
# 
# ( 0.09466/0.10926129027524115 = 86,68%) de l’inertie totale.
# 

# In[86]:


#affichage graphique des v.p.
afc.plot_eigenvalues()
print(numpy.sum(poidsLig* afc.row_coord_[:,1]**2))


# Nous représentons les individus lignes dans le premier plan factoriel qui restitue 97.3% de l’information disponible.

# In[87]:


#affichage dans le premier plan factoriel
fig, ax = plt.subplots(figsize=(10,10))
ax.axis([-0.5,+0.5,-0.5,+0.5])
ax.plot([-0.5,+0.5],[0,0],color='red',linestyle='--')
ax.plot([0,0],[-0.5,+0.5],color='red',linestyle='--')
ax.set_xlabel("Dim.1 (86,68% )")
ax.set_ylabel("Dim.2 (10,62%)")
plt.title("Carte des modalités lignes")
for i in range(D.shape[0]):
 ax.text(afc.row_coord_[i,0],afc.row_coord_[i,1],D.index[i])

plt.show()


# En mettant la même échelle en abscisse et ordonnées, nous constatons que la différenciation des modalités lignes se joue 
# 
# quasi-exclusivement sur le premier axe factoriel

# Calculons les distances euclidienne dans le 1er plan.

# In[88]:


distPairesLigF1 = numpy.zeros(shape=(prof_lig.shape[0],prof_lig.shape[0]))
#double boucle
for i in range(prof_lig.shape[0]-1):
   for j in range(0,prof_lig.shape[0]):
      distPairesLigF1[i,j] = numpy.sum((afc.row_coord_[i,0]-afc.row_coord_[j,0])**2)

#affichage
print(pandas.DataFrame(distPairesLigF1,index=D.index,columns=D.index))


# # Analyse des colonnes.

#  De la même manière que pour les profils lignes, nous pouvons calculer les distances à l’origine , les poids (des colonnes) et les inerties.

# Nous calculons tout d’abord le profil « moyen » des filières.

# In[89]:


#profil marginal des filières
prof_marg_col = tot_lig/numpy.sum(tot_lig)
print(prof_marg_col)


# In[90]:


#tableau des profils colonnes
prof_col = numpy.apply_along_axis(arr=D.values,axis=0,func1d=lambda x:x/numpy.sum(x))
print(pandas.DataFrame(prof_col,index=D.index,columns=D.columns))


# Pour chaque colones, nous formons la distance à l’origine

# In[91]:


#distance**2 à l'orgine
distoCol = numpy.apply_along_axis(arr=prof_col,axis=0,func1d=lambda x:numpy.sum((x
-prof_marg_col)**2/prof_marg_col))

#affichage
print(pandas.DataFrame(distoCol,index=D.columns))


# La composition des opinions est manifestement différente de la globalité. En comparant son profil avec le profil marginal, nous notons une surreprésentation des opinion des groupes d'âge 25-34, 35-44 et 45-54  , et une sous-représentation des les vieillards.
# 

# calculons le poids 

# In[92]:


#poids de chaque colonne
poidsCol = tot_col/numpy.sum(tot_col)
print(pandas.DataFrame(poidsCol,index=D.columns))


# calculons les inerties.

# In[93]:


#inertie
inertieCol = distoCol*poidsCol
print(pandas.DataFrame(inertieCol,index=D.columns))


# Les 
# jugement "Bon" et "Mauvais" sont celles qui pèseront le plus dans l’analyse, mais pas pour les mêmes raisons : la première"BON"
# parce que sa distance à l'origine est élevée 0.901452 , la seconde parce qu’elle rassemble un grand nombre des opinions (poids  = 0.599853 ).

# In[94]:


#somme des inerties
print(numpy.sum(inertieCol))


#  La somme des inerties des modalités colonnes identique à la somme des inerties des modalités lignes . Ce n’est pas un       hasard.
# 
# En AFC, les lignes et les colonnes jouent des roles symetriques(dualité). 

# nous affichons les coordonnées des modalités colonnes pour les 3 facteurs de l’AFC sur les données .

# In[95]:



print(pandas.DataFrame(afc.col_coord_,index=D.columns))


# passons à la représentation graphique dans le plan.

# In[96]:



#affichage dans le premier plan factoriel
fig, ax = plt.subplots(figsize=(10,10))
ax.axis([-0.7,+0.7,-0.7,+0.7])
ax.plot([-0.7,+0.7],[0,0],color='blue',linestyle='--')
ax.plot([0,0],[-0.7,+0.7],color='blue',linestyle='--')
ax.set_xlabel("Dim.1 ")
ax.set_ylabel("Dim.2 ")
plt.title("Carte des modalités colonnes")

for i in range(D.shape[1]):
   ax.text(afc.col_coord_[i,0],afc.col_coord_[i,1],D.columns[i])
plt.show()


# # Analyse de l’association lignes-colonnes

# nous formons le tableau des effectifs théoriques :

# In[97]:


#effectifs totaux
n = numpy.sum(D.values)
#tableau sous indépendance
E = numpy.dot(numpy.reshape(tot_lig,(7,1)),numpy.reshape(tot_col,(1,4)))/n
print(E)


# Nous pouvons calculer la statistique de test et la probabilité critique :

# In[98]:


#statistique du KHI-2
KHI2 = numpy.sum(((D.values-E)**2)/E)
print(KHI2)


# In[99]:


#degré de liberté
ddl = (E.shape[0]-1)*(E.shape[1]-1)
print(ddl)


# La valeur p du test.

# In[100]:


print(1-scipy.stats.chi2.cdf(KHI2,ddl))


# Si la valeur p est inférieure à 0,05. Cela indique des preuves solides contre l'hypothèse nulle, car il y a moins de 5% de probabilité que la valeur nulle soit correcte (et les résultats sont aléatoires). Par conséquent, nous rejetons l'hypothèse nulle.
# 
# Sinon nous conservons l'hypothèse nulle.
# 
# Donc, Le test conduit au rejet de l’hypothèse nulle (p-value ≈ 0). Manifestement
# 

# In[101]:


#représentation simultanée
fig, ax = plt.subplots(figsize=(10,10))
ax.axis([-0.7,+0.7,-0.7,+0.7])
ax.plot([-0.7,+0.7],[0,0],color='silver',linestyle='--')
ax.plot([0,0],[-0.7,+0.7],color='silver',linestyle='--')
ax.set_xlabel("Dim.1 (97.35% )")
ax.set_ylabel("Dim.2 (2.01%)")
plt.title("Carte des modalités lignes et colonnes")
#modalités ligne
for i in range(D.shape[0]):
 ax.text(afc.row_coord_[i,0],afc.row_coord_[i,1],D.index[i],color='blue')
#modalités colonne
for i in range(D.shape[1]):
 ax.text(afc.col_coord_[i,0],afc.col_coord_[i,1],D.columns[i],color='red')

plt.show()

