'''
Yashwant Datti & Sathvik Chilakala
CSE 163 Final Project
Research Question 1
'''

# All Importations
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

bread_prices = pd.read_csv('datasets/bread_price_spain.csv')
inflation_rates = pd.read_csv('datasets/inflation_rate_spain.csv')
gdp_per_capita = pd.read_csv('datasets/gdp_per_capita_spain.csv')
unemployment_rates = pd.read_csv('datasets/unemployment_rate_spain.csv')

def inflationvsprice():
    '''
    
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    merged_price = bread_prices.merge(inflation_rates, left_on='date', right_on='date', how='inner')
    merged_rate = inflation_rates.merge(bread_prices, left_on='date', right_on='date', how='inner')
    merged_price['date'] = pd.to_datetime(merged_price['date'])
    merged_price['date'] = merged_price['date'].dt.year * 10000 + merged_price['date'].dt.month * 100 + merged_price['date'].dt.day
    x_var = merged_rate['Inflation']
    y_var = merged_price['date']
    z_var = merged_price['COST']
    ax.scatter3D(x_var, y_var, z_var, 'blue')
    ax.plot3D(x_var, y_var, z_var, 'blue')
    ax.set_title('Inflation vs. Bread Price in Spain')
    ax.set_xlabel('Inflaton Rate (%)')
    ax.set_ylabel('Date')
    ax.set_zlabel('Price (Spanish Peseta)')
    ax.view_init(elev=30, azim=45)
    fig.savefig('InflationVsCostSpain.png')

def gdpvsprice():
    merged_price = bread_prices.merge(gdp_per_capita, left_on='date', right_on='date', how='inner')
    merged_rate = gdp_per_capita.merge(bread_prices, left_on='date', right_on='date', how='inner')
    print(merged_price)
    print(merged_rate)
    GDP, COST = np.polyfit(merged_rate['GDP'],merged_price['COST'], 1)
    plt.figure(figsize=(10,6))
    plt.scatter(merged_rate['GDP'],merged_price['COST'])
    plt.plot(merged_rate['GDP'], GDP*merged_rate['GDP']+COST)
    plt.title('GDP Per Capita vs. Bread Price in Spain')
    plt.xlabel('GDP Per Capita')
    plt.ylabel('Price (Spanish Peseta)')
    plt.grid(True)
    plt.savefig('GDPvsCostSpain.png')
    

def unemploymentvsprice():
    merged_price = bread_prices.merge(unemployment_rates, left_on='date', right_on='date', how='inner')
    merged_rate = unemployment_rates.merge(bread_prices, left_on='date', right_on='date', how='inner')
    print(merged_price)
    print(merged_rate)
    plt.figure(figsize=(10,6))
    plt.bar(merged_rate['Unemployment'], merged_price['COST'])
    plt.scatter(merged_rate['Unemployment'], merged_price['COST'])
    plt.title('Unemployment vs. Bread Price in Spain')
    plt.xlabel('Unemployment Rate (%)')
    plt.ylabel('Price (Spanish Peseta)')
    plt.grid(True)
    plt.savefig('UnemploymentVsCostSpain.png')


def main():
    inflationvsprice()
    unemploymentvsprice()
    gdpvsprice()

if __name__ == '__main__':
    main()