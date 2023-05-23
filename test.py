# All Importations
import question0
import question1
import question2 
import question2
import pandas as pd
from cse163_utils import assert_equals, compare_plots


def test_load_data():
    data = question0.load_data()
    assert_equals(['Year', 'Average price', 'Inflation adjusted price (2023 dollars)'], list(data.columns))

def test_load_and_merge():
    """
    This method tests the merged datasets
    within each question file. 
    """
    global_price_file = 'test_datasets/sample_bread_price_global.csv'
    world_json_file = 'test_datasets/sample_world.json'
    world_data, merged_data = question0.load_and_merge(global_price_file, world_json_file)
    assert_equals(['id', 'name', 'geometry', 'Country_Name', 'Amount', 'year'], list(merged_data.columns))


def check_plots():
    assert_equals(True, compare_plots('plots/InflationVsCostSpain.png', 'expected/InflationVsCostSpain.png'))
    assert_equals(True, compare_plots('plots/InflationChangeVsCost.png', 'expected/InflationChangeVsCost.png'))
    assert_equals(True, compare_plots('plots/GDPvsCostSpain.png', 'expected/GDPvsCostSpain.png'))
    assert_equals(True, compare_plots('plots/UnemploymentVsCostSpain.png', 'expected/UnemploymentVsCostSpain.png'))
    assert_equals(True, compare_plots('plots/Global_Bread_Prices.png', 'expected/Global_Bread_Prices.png'))
    assert_equals(True, compare_plots('plots/Bread_Price_Over_Intervals.png', 'expected/Bread_Price_Over_Intervals.png'))


"""
Begin Testing Machine Learning Model
"""
predictor = question2.BreadPredictor()
BREAD_PRICE, GDP, IMPORTS, INFLATION, UNEMPLOYMENT, WAGE, EXPORTS = predictor.load_data()

def test_ML_load_data():
    assert_equals(BREAD_PRICE.shape, (759, 2))
    assert_equals(GDP.shape, (759, 2))
    assert_equals(IMPORTS.shape, (759, 2))
    assert_equals(INFLATION.shape, (759, 2))
    assert_equals(UNEMPLOYMENT.shape, (759, 2))
    assert_equals(WAGE.shape, (759, 2))
    assert_equals(EXPORTS.shape, (759, 2))

def test_preprocessed_data():
    '''
    This method tests the preprocesses data
    that the machine learning model used to
    train itself with. 
    '''
    x_train, x_test, y_train, y_test = predictor.preprocess_data(BREAD_PRICE, GDP, IMPORTS, INFLATION, UNEMPLOYMENT, WAGE, EXPORTS)
    assert_equals(x_train.shape, (683, 6))
    assert_equals(x_test.shape, (76, 6))
    assert_equals(y_train.shape, (683, 1))
    assert_equals(y_test.shape, (76, 1))



def main():
    test_load_data()
    test_load_and_merge()
    test_ML_load_data()
    test_preprocessed_data()
    check_plots()


if __name__ == '__main__':
    main()


