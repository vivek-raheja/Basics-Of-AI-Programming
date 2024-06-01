#Vivek Raheja
#ITP 259 Spring 2024
#HW2

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer

def main():
    wineData = pd.read_csv("wineQualityReds.csv")
    pd.set_option("display.max_columns", None)
    wineData.drop('Wine', axis=1, inplace=True) # drop wine
    quality = wineData['quality'] # extract quality
    wineData.drop('quality', axis=1, inplace=True) # drop quality
    print(quality)
    print(wineData) # print quality and dataframe


    norm = Normalizer()
    wineData_norm = pd.DataFrame(norm.transform(wineData), columns=wineData.columns)
    print(wineData_norm) # print normalized dataframe

    ks = range(1, 11) #KMeans clustering
    inertias = []
    for k in ks:
        model = KMeans(n_clusters=k)
        model.fit(wineData_norm)
        inertias.append(model.inertia_)

    # inertias vs. number of clusters
    plt.plot(ks, inertias, "-o")
    plt.xlabel("Number of Clusters, k")
    plt.ylabel("Inertia")
    plt.xticks(ks)
    plt.show()

    model = KMeans(n_clusters=6, random_state=2023) # picked k = 6 using elbow method
    model.fit(wineData_norm)
    labels = model.predict(wineData_norm)
    wineData_norm['cluster'] = pd.Series(labels)


    wineData_norm['quality'] = quality # adding quality column back in
    crosstab = pd.crosstab(wineData_norm['quality'], wineData_norm['cluster'])
    print(crosstab)

    # Our crosstab shows that wines of all qualities are present in each cluster,
    # meaning the clusters do not accurately represent the quality of the wines.

    # However, Clusters 4 and 5 show the highest concentration of wines of quality 5 and 6.
    # This could be the clusters identifying the average quality of wines that are most prevalent in the data set.


main()