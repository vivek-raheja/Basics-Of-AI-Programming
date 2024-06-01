#Vivek Raheja
#ITP 259 Spring 2024
#HW1

import pandas as pd

def main():
    # Question 1:
    wineData = pd.read_csv("wineQualityReds.csv")
    print(wineData)


    # Question 2:
    wineData = pd.read_csv("wineQualityReds.csv")
    print(wineData.head(10))


    # Question 3:
    wineData = pd.read_csv("wineQualityReds.csv")
    sorted_wineData = wineData.sort_values(by='volatile.acidity', ascending=False)
    print(sorted_wineData)


    # Question 4:
    wineData = pd.read_csv("wineQualityReds.csv")
    seven_wines = wineData[wineData['quality'] == 7]
    print(seven_wines)


    # Question 5:
    wineData = pd.read_csv("wineQualityReds.csv")
    average_pH = wineData['pH'].mean()
    print(average_pH)


    # Question 6:
    wineData = pd.read_csv("wineQualityReds.csv")
    alcohol_wines_ten = wineData[wineData['alcohol'] > 10]
    print(len(alcohol_wines_ten))


    # Question 7:
    wineData = pd.read_csv("wineQualityReds.csv")
    max_alcohol = wineData['alcohol'].max()
    most_alcoholic_wine = wineData[wineData['alcohol'] == max_alcohol]
    print(most_alcoholic_wine)


    # Question 8:
    wineData = pd.read_csv("wineQualityReds.csv")
    random_wine = wineData.sample(n=1)
    residual_sugar_level = random_wine['residual.sugar'].iloc[0]
    print(residual_sugar_level)


    # Question 9:
    wineData = pd.read_csv("wineQualityReds.csv")
    quality_4_wines = wineData[wineData['quality'] == 4]
    random_quality_4_wine = quality_4_wines.sample(n=1)
    print(random_quality_4_wine)


    # Question 10:
    wineData = pd.read_csv("../raheja_vivek_hw2/wineQualityReds.csv")
    wines_not_four = wineData[wineData['quality'] != 4]
    print(len(wines_not_four))



main()