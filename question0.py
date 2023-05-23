'''
Yashwant Datti & Sathvik Chilakala
CSE 163 Final Project
Research Question 0
'''


# All Importations
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def load_data():
    '''
    This method loads in the data from the csv file
    1980-2016_bread_price_average and returns the 
    dataframe.
    '''
    df = pd.read_csv('datasets/1980-2016_bread_price_average.csv')
    return df


def show_time_periods(data):
    '''
    This method returns a lineplot that displays the price
    of bread over the past thirty-six years from 1980-2016.
    On the graph shows multiple time intervals in different
    colors to better display areas of growth. The plot is
    saved to Bread_Price_Over_Intervals.png.
    '''
    data['Year'] = pd.to_datetime(data['Year'], format='%Y')
    time_periods = [
        {'start': 1980, 'end': 1989},
        {'start': 1989, 'end': 1991},
        {'start': 1991, 'end': 1994},
        {'start': 1994, 'end': 2000},
        {'start': 2000, 'end': 2015},
        {'start': 2015, 'end': 2016},
    ]
    for period in time_periods:
        period_data = data[(data['Year'] >= str(period['start'])) & (data['Year'] <= str(period['end']))]
        plt.plot(period_data['Year'], period_data['Average price'], label=f"{period['start']}-{period['end']}")
    plt.xlabel('Year') 
    plt.ylabel('Price of Bread')
    plt.legend()
    plt.savefig('plots/Bread_Price_Over_Intervals.png')


def load_and_merge(global_price, world_json):
    '''
    This function loads two files and merges
    both of them so that it can be easily plotted
    using the module Geopandas. 
    '''
    global_data = pd.read_csv(global_price)
    world_data = gpd.read_file(world_json)

    merged_country_name = world_data.merge(global_data, left_on='name', right_on='Country_Name', how='inner')
    merged_country_name['Amount'] = merged_country_name['Amount'].astype(float)
    return (world_data, merged_country_name)


def plot_data(world_data, merged_data):
    '''
    This function takes the merged data and plots
    it using geopandas. It plots the bread price
    per country on a global map. 
    '''
    fig, axs = plt.subplots(1, figsize=(10,10))
    world_data.plot(ax=axs, color='#CCCCCC')
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    merged_data.plot(ax=axs, column='Amount', legend=True, cax=cax)
    axs.set_title('Bread Prices around the Globe')
    fig.savefig('plots/Global_Bread_Prices.png', bbox_inches='tight')


    

def main():
    data = load_data()
    show_time_periods(data)

    world_data, merged_data = load_and_merge('datasets/bread_price_global.csv', 'datasets/world.json')
    plot_data(world_data, merged_data)


if __name__ == "__main__":
    main()
