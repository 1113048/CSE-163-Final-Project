'''
Yashwant Datti & Sathvik Chilakala
CSE 163 Final Project
Research Question 1
'''

# All Importations
from random import choice
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

bread_prices = pd.read_csv('datasets/bread_price_spain.csv')
inflation_rates = pd.read_csv('datasets/inflation_rate_spain.csv')
gdp_per_capita = pd.read_csv('datasets/gdp_per_capita_spain.csv')
unemployment_rates = pd.read_csv('datasets/unemployment_rate_spain.csv')

def inflation_vs_price():
    '''
    This method plots on a 3-D line plot the correlation
    between inflation, time, and the price of bread in
    the country of Spain.
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
    # Plotting Data
    ax.scatter3D(x_var, y_var, z_var, color='chocolate')
    ax.set_title('Inflation vs. Bread Price in Spain')
    ax.set_xlabel('Inflaton Rate (%)')
    ax.set_ylabel('Date')
    ax.set_zlabel('Price (Spanish Peseta)')
    ax.view_init(elev=30, azim=45)
    ax.legend(['Price Growth'])
    fig.savefig('plots/Inflation_Vs_Cost_Spain.png')

def inflation_change_vs_price():
    '''
    This method plots on a line plot the rate of
    change in inflation in comparison to the
    price of bread in the country of Spain.
    '''
    merged_price = bread_prices.merge(inflation_rates, left_on='date', right_on='date', how='inner')
    merged_rate = inflation_rates.merge(bread_prices, left_on='date', right_on='date', how='inner')
    plt.figure(figsize=(10,6))
    plt.step(merged_rate['Change'],merged_price['COST'], color='chocolate')
    # Plotting Data
    plt.title('Inflation Rate Change vs. Bread Price in Spain')
    plt.xlabel('Inflation Rate Change Yearly')
    plt.ylabel('Price (Spanish Peseta)')
    plt.legend('Price Growth')
    plt.savefig('plots/Inflation_Change_Vs_Cost.png')


def gdp_vs_price():
    '''
    This method plots a scatter plot as well
    as a line of best fit. This plot compares
    GDP per capita and the price of bread in
    the country of Spain.
    '''
    merged_price = bread_prices.merge(gdp_per_capita, left_on='date', right_on='date', how='inner')
    merged_rate = gdp_per_capita.merge(bread_prices, left_on='date', right_on='date', how='inner')
    GDP, COST = np.polyfit(merged_rate['GDP'],merged_price['COST'], 1)
    plt.figure(figsize=(10,6))
    plt.scatter(merged_rate['GDP'],merged_price['COST'], color='peru')
    # Plotting Data
    plt.plot(merged_rate['GDP'], GDP*merged_rate['GDP']+COST, color='chocolate')
    plt.title('GDP Per Capita vs. Bread Price in Spain')
    plt.xlabel('GDP Per Capita')
    plt.ylabel('Price (Spanish Peseta)')
    plt.grid(True)
    plt.savefig('plots/GDP_vs_Cost_Spain.png')
    

def unemployment_vs_price():
    '''
    This method plots a bar chart that compares
    the unemployment rates to the price of
    bread in the country of Spain.
    '''
    merged_price = bread_prices.merge(unemployment_rates, left_on='date', right_on='date', how='inner')
    merged_rate = unemployment_rates.merge(bread_prices, left_on='date', right_on='date', how='inner')
    plt.figure(figsize=(10,6))
    # Plotting Data
    plt.bar(merged_rate['Unemployment'], merged_price['COST'], color='chocolate')
    plt.scatter(merged_rate['Unemployment'], merged_price['COST'], color='peru')
    plt.title('Unemployment vs. Bread Price in Spain')
    plt.xlabel('Unemployment Rate (%)')
    plt.ylabel('Price (Spanish Peseta)')
    plt.grid(True)
    plt.savefig('plots/Unemployment_Vs_Cost_Spain.png')



def main():
    inflation_vs_price()
    unemployment_vs_price()
    inflation_change_vs_price()
    gdp_vs_price()

if __name__ == '__main__':
    main()