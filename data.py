#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.pipeline
import sklearn.base
import sklearn.linear_model
import sklearn.preprocessing
import matplotlib.pyplot as plt
import math

class DerivedAttributesAdder( sklearn.base.BaseEstimator, sklearn.base.TransformerMixin ):
    def __init__( self ):
        return

    def fit( self, X, y=None ):
        return self

    def transform( self, X, y=None ):
        # insert x5 squared as a feature
        X[ 'x5^2' ] = pd.Series( X[ 'x5' ] * X[ 'x5' ] )
        # X[ 'x5^3' ] = pd.Series( X[ 'x5' ] * X[ 'x5' ] * X[ 'x5' ]  )
        # X[ 'x5^4' ] = pd.Series( X[ 'x5' ] * X[ 'x5' ] * X[ 'x5' ] * X[ 'x5' ] )
        # X[ 'log(x5)' ] = pd.Series( np.log( X[ 'x5' ] ) )
        X = X.drop( [ 'x5',  ], axis=1 )
        return X
    
# keep predictors, remove labels
class DataFrameSelector( sklearn.base.BaseEstimator, sklearn.base.TransformerMixin ):
    
    def __init__( self, do_predictors=True ):
        self.mPredictors = [ "x1", "x2", "x3", "x4", "x5", "x6", "x7" ]
        self.mLabels = [ "labels" ]
        self.mDoPredictors = do_predictors
        return

    def fit( self, X, y=None ):
        # no fit necessory
        return self

    def transform( self, X, y=None ):
        if self.mDoPredictors:
            values = X[ self.mPredictors ]
        else:
            values = X[ self.mLabels ]
            
        # print( "DataFrameSelector" )
        # print( values )
        # print( "-----------" )
        
        return values

class OutlierCuts( sklearn.base.BaseEstimator, sklearn.base.TransformerMixin ):

    def __init__( self ):
        return

    def fit( self, X, y=None ):
        # do nothing
        return self

    def transform( self, X, y=None ):
        # values = # the transformed values of X
        values = X[ ( X.x1 >= 35 ) & ( X.x1 <= 145 ) ]
        values = values[ ( values.x2 >= 64 ) & ( values.x2 <= 70 ) ]
        values = values[ ( values.x3 >= 40 ) & ( values.x3 <= 75 ) ]
        values = values[ ( values.x4 >= 25 ) & ( values.x4 <= 115 ) ]
        values = values[ ( values.x5 >= 0 ) & ( values.x5 <= 170 ) ]
        values = values[ ( values.x6 >= 20 ) & ( values.x6 <= 60 ) ]
        values = values[ ( values.x7 >= 26 ) & ( values.x7 <= 26.6 ) ]
        #values = values[ ( values.labels >= -10 ) & ( values.labels <= 200000 ) ]
        # print( "OutlierCuts" )
        # print( values )
        # print( "-----------" )
        return values

def make_predictor_pipeline( ):
    items = [ ]
    items.append( ( "remove-outliers", OutlierCuts( ) )  )
    items.append( ( "predictors-only", DataFrameSelector( True ) )  )
    items.append( ( "derived-attributes", DerivedAttributesAdder( ) )  )
    # Note that StandardScaler will transform data from pandas.DataFrame to numpy.array
    items.append( ( "scaler", sklearn.preprocessing.StandardScaler( copy=False ) ) )
    pipeline = sklearn.pipeline.Pipeline( items )
    return pipeline

def make_label_pipeline( ):
    items = [ ]
    items.append( ( "remove-outliers", OutlierCuts( ) )  )
    items.append( ( "labels-only", DataFrameSelector( False ) )  )
    pipeline = sklearn.pipeline.Pipeline( items )
    return pipeline

def get_data( filename ):
    data = pd.read_csv( filename, index_col=0 )
    return data

def split_data( data, ratio ):
    data_train, data_test = sklearn.model_selection.train_test_split( data, test_size=ratio )
    data_train.to_csv( "data-train.csv" )
    data_test.to_csv( "data-test.csv" )
    return

def display_histograms( predictor_data, label_data, basename ):
    # predictor_data is a numpy.array with 7 features
    # label_data is a pandas.DataFram with 1 column, "labels"
    columns = predictor_data.shape[ 1 ]
    sp_rows = math.ceil( math.sqrt( columns ) )
    sp_cols = math.ceil( math.sqrt( columns ) )
    
    plt.suptitle( "Feature Histograms" )
    fig_num = 2
    plt.figure( fig_num, figsize=(6.5, 9) )
    for i in range( 1, columns+1 ):
        name = 'x' + str( i )
        plt.subplot( sp_rows, sp_cols, i )
        #plt.yscale( "log" )
        plt.hist( predictor_data[ :, i-1 ], bins=20 )
        plt.xlabel( name )
        plt.locator_params( axis='x', tight=True, nbins=5 )

    plt.subplot( sp_rows, sp_cols, i+1 )
    plt.yscale( "log" )
    plt.hist( label_data[ 'labels' ], bins=20 )
    plt.xlabel( 'labels' )
    plt.locator_params( axis='x', tight=True, nbins=5 )

    plt.tight_layout( )
    plt.savefig( basename + '-histogram.pdf' )
    #plt.show( )
    
    return

# def create_square_data( X ):
#     X2 = X * X * 50000
#     return X2

def display_slopes( predictor_data, label_data, basename ):
    # predictor_data is a numpy.array with 7 features
    # label_data is a pandas.DataFram with 1 column, "labels"
    columns = predictor_data.shape[ 1 ]
    sp_rows = math.ceil( math.sqrt( columns ) )
    sp_cols = math.ceil( math.sqrt( columns ) )
    plt.suptitle( "Labels vs. Features" )
    
    plt.figure( 1, figsize=(6.5, 9) )
    for i in range( 1, columns+1 ):
        name = 'x' + str( i )
        plt.subplot( sp_rows, sp_cols, i )
        X = predictor_data[ :, i-1 ]
        #X2 = create_square_data( X )
        Y = label_data[ 'labels' ]
        plt.scatter( X, Y, s=1, color='blue' )
        #plt.scatter( X, X2, s=1, color='green' )
        plt.xlabel( name )
        plt.ylabel( 'labels' )
        plt.locator_params( axis='both', tight=True, nbins=5 )

    plt.subplot( sp_rows, sp_cols, i+1 )
    plt.scatter( 'labels', 'labels', data=label_data, s=1, color='blue' )
    plt.xlabel( 'labels' )
    plt.ylabel( 'labels' )
    plt.locator_params( axis='both', tight=True, nbins=5 )

    plt.tight_layout( )
    plt.savefig( basename + '-slopes.pdf' )
    #plt.show( )

    return

def display_predicted_slopes( predictor_data, label_data, predicted_label_data, basename ):
    # predictor_data is a numpy.array with 7 features
    # label_data is a pandas.DataFrame with 1 column, "labels"
    # predicted_label_data is an array with 1 column, the labels predicted
    columns = predictor_data.shape[ 1 ]
    sp_rows = math.ceil( math.sqrt( columns ) )
    sp_cols = math.ceil( math.sqrt( columns ) )
    plt.suptitle( "Predicted Labels vs. Features" )
    
    plt.figure( num=None, figsize=(6.5, 9) )
    for i in range( 1, columns+1 ):
        name = 'x' + str( i )
        plt.subplot( sp_rows, sp_cols, i )
        X = predictor_data[ :, i-1 ]
        Y = label_data[ 'labels' ]
        plt.scatter( X, Y, s=0.5, color='blue' )
        
        Y = predicted_label_data
        plt.scatter( X, Y, s=0.5, color='red' )
        
        plt.xlabel( name )
        plt.ylabel( 'predicted labels' )
        plt.locator_params( axis='both', tight=True, nbins=5 )

    plt.subplot( sp_rows, sp_cols, i+1 )

    plt.scatter( 'labels', 'labels', data=label_data, s=1, color='blue' )
    plt.scatter( predicted_label_data, predicted_label_data, s=1, color='red' ) 
    
    plt.xlabel( 'predicted labels' )
    plt.ylabel( 'predicted labels' )
    plt.locator_params( axis='both', tight=True, nbins=5 )

    plt.tight_layout( )
    plt.savefig( basename + '-predicted_slopes.pdf' )
    #plt.show( )

    return

def display_data(  predictor_data, label_data, basename ):
    display_slopes(  predictor_data, label_data, basename )
    display_histograms(  predictor_data, label_data, basename )
    return

def display_predicted_data(  predictor_data, label_data, predicted_label_data, basename ):
    display_predicted_slopes(  predictor_data, label_data, predicted_label_data, basename )
    return

def main( ):
    do_split = False
    do_plots = True
    do_fit = True
    do_test = True
    

    if do_split:
        data = get_data( "data.csv" )
        split_data( data, 0.20 )
    else:
        data = get_data( "data-train.csv" )
        predictor_pipeline = make_predictor_pipeline( )
        label_pipeline = make_label_pipeline( )
        predictors_processed = predictor_pipeline.fit_transform( data )
        labels_processed = label_pipeline.fit_transform( data )
        
    if do_plots:
        display_data( predictors_processed, labels_processed, "data" )
        
    if do_fit:
        linear_regression = sklearn.linear_model.LinearRegression( )
        linear_regression.fit( predictors_processed, labels_processed )

        print( linear_regression.coef_ )
        print( linear_regression.intercept_ )

    if do_fit and do_plots:
        training_labels_predicted = linear_regression.predict( predictors_processed )
        display_predicted_data( predictors_processed, labels_processed, training_labels_predicted, "data-predicted" )

    if do_fit and do_test:
        test_data = get_data( "data-test.csv" )
        test_predictors_processed = predictor_pipeline.transform( test_data )
        test_labels_processed = label_pipeline.transform( test_data )
        test_labels_predicted = linear_regression.predict( test_predictors_processed )

        mean_squared_error = sklearn.metrics.mean_squared_error( test_labels_processed, test_labels_predicted )
        root_mean_squared_error = np.sqrt( mean_squared_error )
        print( "Error: " + str( root_mean_squared_error ) )
        
        if do_fit and do_test and do_plots:
            display_predicted_data( test_predictors_processed, test_labels_processed, test_labels_predicted, "data-test" )
        
    return

if __name__ == "__main__":
    main( )
