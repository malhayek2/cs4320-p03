#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.metrics

def get_data( filename ):
    data = pd.read_csv( filename, index_col=0 )
    return data

def separate_predictors_and_labels( data ):
    predictors_X = data.drop( "labels", axis=1 )
    labels_Y = data[ "labels" ].copy( )
    return predictors_X, labels_Y

def scale_predictors( X ):
    X = X.astype( 'float64' )
    scaler = sklearn.preprocessing.StandardScaler( )
    scaler.fit( X )
    X = scaler.transform( X )
    return X, scaler

def fit( X, Y ):
    reg = sklearn.linear_model.LinearRegression( )
    reg.fit( X, Y )
    print( )
    print( reg )
    print( reg.coef_ )
    print( reg.intercept_ )
    return reg

def test( X, Y, reg ):
    Y_predicted = reg.predict( X )
    print( Y )
    print( Y_predicted )

    mean_squared_error = sklearn.metrics.mean_squared_error( Y, Y_predicted )
    root_mean_squared_error = np.sqrt( mean_squared_error )
    print( "Error: " + str( root_mean_squared_error ) )
    return

def main( ):
    # np.random.seed( 42 )
    data_train = get_data( "data-train.csv" )
    train_X_raw, train_Y = separate_predictors_and_labels( data_train )
    train_X, scaler = scale_predictors( train_X_raw )
    print( train_X_raw )
    print( train_X )
    print( scaler )
    reg = fit( train_X, train_Y )
    
    data_test = get_data( "data-test.csv" )
    test_X_raw, test_Y = separate_predictors_and_labels( data_test )
    test_X_raw = test_X_raw.astype( 'float64' )
    test_X = scaler.transform( test_X_raw )

    test( test_X, test_Y, reg )
    
    return

if __name__ == "__main__":
    main( )
    
