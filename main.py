import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.metrics
from sklearn.externals import joblib
import sklearn.pipeline
import math
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from time import time

#open data file
def get_data( filename ):
	print("given filename: ", filename)
	#we have added index_col to none to not ignore any col unlike first assignment
	data = pd.read_csv( filename, index_col=None )
	#print(data.columns.values)
	return data

#display data 
def display_data_raw( data ):
	#setting figure 1 
	plt.figure( 1, figsize=(9,6) )
	#subplot in 3Fly_Ash grid, position 1
	plt.subplot(4, 4, 1 )
	plt.scatter( data.Cement,data.Concrete_compressive_strength)
	plt.title('Cement Scatter')
    # subplot in 3Fly_Ash grid, position 2
	plt.subplot( 4, 4, 2 )
	plt.bar( data.Cement,data.Concrete_compressive_strength)
	plt.title('Cement Bar')
	
	plt.subplot( 4, 4, 3 )
	plt.scatter( data.Blast_Furnace_Slag,data.Concrete_compressive_strength)
	plt.title('Blast_Furnace_Slag Scatter')
	plt.subplot( 4, 4, 4 )
	plt.bar( data.Blast_Furnace_Slag,data.Concrete_compressive_strength)
	plt.title('Blast_Furnace_Slag Bar')

	plt.subplot( 445 )
	plt.scatter(data.Fly_Ash, data.Concrete_compressive_strength)
	plt.title('Fly_Ash Scatter')
	plt.subplot( 446 )
	plt.bar(data.Fly_Ash, data.Concrete_compressive_strength)
	plt.title('Fly_Ash Bar')

	plt.subplot( 447 )
	plt.scatter( data.water,data.Concrete_compressive_strength)
	plt.title('water Scatter')
	plt.subplot( 448 )
	plt.bar( data.water,data.Concrete_compressive_strength)
	plt.title('water Bar')

	plt.subplot( 449 )
	plt.scatter( data.Superplasticizer,data.Concrete_compressive_strength)
	plt.title('Superplasticizer Scatter')
	plt.subplot( 4,4,10 )
	plt.bar(data.Superplasticizer, data.Concrete_compressive_strength )
	plt.title('Superplasticizer Bar')


	plt.subplot( 4,4,11 )
	plt.scatter( data.Coarse_Aggregate,data.Concrete_compressive_strength)
	plt.title('Coarse_Aggregate Scatter')
	plt.subplot( 4,4,12 )
	plt.bar( data.Coarse_Aggregate,data.Concrete_compressive_strength)
	plt.title('Coarse_Aggregate Bar')

	plt.subplot( 4,4,13 )
	plt.scatter(data.Fine_Aggregate,data.Concrete_compressive_strength)
	plt.title('Fine_Aggregate Scatter')
	plt.subplot( 4,4,14 )
	plt.bar(data.Fine_Aggregate,data.Concrete_compressive_strength)
	plt.title('Fine_Aggregate Bar')


	plt.subplot( 4,4,15 )
	plt.scatter(data.Age,data.Concrete_compressive_strength)
	plt.title('Age Scatter')
	plt.subplot( 4,4,16 )
	plt.bar(data.Age,data.Concrete_compressive_strength)
	plt.title('Age Bar')

	plt.suptitle( "Plotting" )

	#plot adjustment (padding) (top, bot window space, hspace and w spance padding between graphs)
	plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.79,
                    wspace=0.35)
	plt.show( )

# keep predictors, remove labels
# remove one label really (Concrete_compressive_strength)
class DataFrameSelector( sklearn.base.BaseEstimator, sklearn.base.TransformerMixin ):
    
    def __init__( self, do_predictors=True ):
    	self.mPredictors = [ "Cement", "Blast_Furnace_Slag", "Fly_Ash", "water",
    	 "Superplasticizer", "Coarse_Aggregate", "Fine_Aggregate", "Age" ]
    	self.mLabels = [ "Concrete_compressive_strength" ]
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


#generating Cement*Cement col
class DerivedAttributesAdder( sklearn.base.BaseEstimator, sklearn.base.TransformerMixin ):
    def __init__( self ):
        return

    def fit( self, X, y=None ):
        return self

    def transform( self, X, y=None ):
        # insert x5 squared as a feature
        X[ 'Cement^2' ] = pd.Series( X[ 'Cement' ] * X[ 'Cement' ] )
        # X[ 'x5^3' ] = pd.Series( X[ 'x5' ] * X[ 'x5' ] * X[ 'x5' ]  )
        # X[ 'x5^4' ] = pd.Series( X[ 'x5' ] * X[ 'x5' ] * X[ 'x5' ] * X[ 'x5' ] )
        # X[ 'log(x5)' ] = pd.Series( np.log( X[ 'x5' ] ) )
        # X = X.drop( [ 'x5',  ], axis=1 )
        return X


def make_predictor_pipeline( ):
    items = [ ]
    #items.append( ( "remove-outliers", OutlierCuts( ) )  )
    items.append( ( "predictors-only", DataFrameSelector( True ) )  )
    items.append( ( "derived-attributes", DerivedAttributesAdder( ) )  )
    # Note that StandardScaler will transform data from pandas.DataFrame to numpy.array
    items.append( ( "scaler", sklearn.preprocessing.StandardScaler( copy=False ) ) )
    pipeline = sklearn.pipeline.Pipeline( items )
    return pipeline

def make_label_pipeline( ):
    items = [ ]
    #items.append( ( "remove-outliers", OutlierCuts( ) )  )
    items.append( ( "labels-only", DataFrameSelector( False ) )  )
    pipeline = sklearn.pipeline.Pipeline( items )
    return pipeline


def split_data( data, ratio ):
    data_train, data_test = sklearn.model_selection.train_test_split( data, test_size=ratio )
    data_train.to_csv( "train_data.csv" )
    data_test.to_csv( "test_data.csv" )
    return


def separate_predictors_and_Concrete_compressive_strength( data ):
    predictors_X = data.drop( "Concrete_compressive_strength", axis=1 )
    Concrete_compressive_strength_Y = data[ "Concrete_compressive_strength" ].copy( )
    return predictors_X, Concrete_compressive_strength_Y

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
	print("reg")
	print( reg )
	print ("reg.coef")
	print( reg.coef_ )
	print ("intercept_")
	print( reg.intercept_ )
	return reg

def test( X, Y, reg ):
	Y_predicted = reg.predict( X )
	print("Test_Y")
	print( Y )
	print ("Y_predicted")
	print( Y_predicted )

	mean_squared_error = sklearn.metrics.mean_squared_error( Y, Y_predicted )
	print("Mean Squared Errpr " + str(mean_squared_error))
	root_mean_squared_error = np.sqrt( mean_squared_error )
	print( "Root Mean Squared Error: " + str( root_mean_squared_error ) )
	return
#NO NEED TO CLEAN CUT
def cleanData():
	data = get_data("mo.csv")
	data = data[(data.Cement < 225 )&(data.Cement >= 40)]
	data = data[(data.Blast_Furnace_Slag < 50 )&(data.Blast_Furnace_Slag >= 30)]
	#-99 isnt working ?
	data = data[(data.Fly_Ash < 150 )&(data.Fly_Ash > -99)]
	data = data[(data.water < 130 )&(data.water >= 40)]
	data = data[(data.Superplasticizer >= 1)]
	data = data[(data.Coarse_Aggregate < 100 )]
	#Fine_Aggregate looks somewhat clean at this point 
	data = data[ ( data.Concrete_compressive_strength > -2000 ) ]
	#save new clean data
	data.to_csv( 'cut_mo.csv' )
	newData = get_data("cut_mo.csv")
	split_data(newData, 0.20 )
	#display_data(newData)

def save_to_joblib(reg):
	joblib.dump(reg, 'linear.joblib')

def display_slopes( predictor_data, label_data, basename ):
    # predictor_data is a numpy.array with 7 features
    # label_data is a pandas.DataFram with 1 column, "labels"
    columns = predictor_data.shape[ 1 ]
    sp_rows = math.ceil( math.sqrt( columns ) )
    sp_cols = math.ceil( math.sqrt( columns ) )
    plt.suptitle( "Labels vs. Features" )
    features = [ "Cement", "Blast_Furnace_Slag", "Fly_Ash", "water",
    	 "Superplasticizer", "Coarse_Aggregate", "Fine_Aggregate", "Age", "" ]
    plt.figure( 1, figsize=(6.5, 9) )
    for i in range( 1, columns+1 ):
    	#print(features[i-1])
    	name = features [i-1]
    	plt.subplot( sp_rows, sp_cols, i )
    	X = predictor_data[ :, i-1 ]
        #X2 = create_square_data( X )
    	Y = label_data[ 'Concrete_compressive_strength' ]
    	plt.scatter( X, Y, s=1, color='blue' )
        #plt.scatter( X, X2, s=1, color='green' )
    	plt.xlabel( name )
    	plt.ylabel( 'labels' )
    	plt.locator_params( axis='both', tight=True, nbins=5 )

    plt.subplot( sp_rows, sp_cols, i )
    plt.scatter( 'labels', 'labels', data=label_data, s=1, color='blue' )
    plt.xlabel( 'labels' )
    plt.ylabel( 'labels' )
    plt.locator_params( axis='both', tight=True, nbins=5 )

    plt.tight_layout( )
    plt.savefig( basename + '-slopes.pdf' )
    #plt.show( )

    return

def display_histograms( predictor_data, label_data, basename ):
    # predictor_data is a numpy.array with 7 features
    # label_data is a pandas.DataFram with 1 column, "labels"
    columns = predictor_data.shape[ 1 ]
    sp_rows = math.ceil( math.sqrt( columns ) )
    sp_cols = math.ceil( math.sqrt( columns ) )
    
    plt.suptitle( "Feature Histograms" )
    fig_num = 2
    features = [ "Cement", "Blast_Furnace_Slag", "Fly_Ash", "water",
    	 "Superplasticizer", "Coarse_Aggregate", "Fine_Aggregate", "Age", " " ]
    plt.figure( fig_num, figsize=(6.5, 9) )
    for i in range( 1, columns+1 ):
        name = features[i-1]
        plt.subplot( sp_rows, sp_cols, i )
        #plt.yscale( "log" )
        plt.hist( predictor_data[ :, i-1 ], bins=20 )
        plt.xlabel( name )
        plt.locator_params( axis='x', tight=True, nbins=5 )

    plt.subplot( sp_rows, sp_cols, i )
    plt.yscale( "log" )
    plt.hist( label_data[ 'Concrete_compressive_strength' ], bins=20 )
    plt.xlabel( 'labels' )
    plt.locator_params( axis='x', tight=True, nbins=5 )

    plt.tight_layout( )
    plt.savefig( basename + '-histogram.pdf' )
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
    features = [ "Cement", "Blast_Furnace_Slag", "Fly_Ash", "water","Superplasticizer", "Coarse_Aggregate", "Fine_Aggregate", "Age", " " ]
    plt.figure( num=None, figsize=(6.5, 9) )
    for i in range( 1, columns+1 ):
        name = features[i-1]
        plt.subplot( sp_rows, sp_cols, i )
        X = predictor_data[ :, i-1 ]
        Y = label_data[ 'Concrete_compressive_strength' ]
        plt.scatter( X, Y, s=0.5, color='blue' )
        
        Y = predicted_label_data
        plt.scatter( X, Y, s=0.5, color='red' )
        
        plt.xlabel( name )
        plt.ylabel( 'predicted labels' )
        plt.locator_params( axis='both', tight=True, nbins=5 )

    plt.subplot( sp_rows, sp_cols, i )

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
	do_plot = True
	do_fit = True
	print("Reading data...")
	#numpy show
	raw_data = get_data("data.csv")

	#display_data_raw(raw_data)
	#splitting data : train data & test data
	split_data(raw_data, 0.20 )
	#reading generated data
	train_data = get_data("train_data.csv")
	test_data = get_data("test_data.csv")
    # data = get_data( "data-train.csv" )
    #now geting our piplines of L & P 
	predictor_pipeline = make_predictor_pipeline( )
	label_pipeline = make_label_pipeline( )
	predictors_processed = predictor_pipeline.fit_transform( train_data )
	labels_processed = label_pipeline.fit_transform( train_data )

	if do_plot:
		display_data( predictors_processed, labels_processed, "data" )
		#display_data(train_X, labels_processed, "train_X")
	# Estimator doing linear regression from processed data. ! 
	if do_fit:
		linear_regression = sklearn.linear_model.LinearRegression( )
		linear_regression.fit( predictors_processed, labels_processed )

		print("LinearR Coeffients : " , linear_regression.coef_ )
		print( "LinearR intercepts: ", linear_regression.intercept_ )

	if do_fit and do_plot:
		#housing_predi
		training_labels_predicted = linear_regression.predict( predictors_processed )
		display_predicted_data( predictors_processed, labels_processed, training_labels_predicted, "data-predicted" )
	

	mean_squared_error = sklearn.metrics.mean_squared_error( labels_processed, training_labels_predicted )
	root_mean_squared_error = np.sqrt( mean_squared_error )
	print( "Error: " + str( root_mean_squared_error ) )

	#(P,train_Data, L.fit_transform(train_data), kfold,scoring="neg_mean_squre")
	cross_valScore_train = cross_val_score( linear_regression,  predictors_processed ,labels_processed,cv=10, scoring = 'neg_mean_squared_error')
	# print(valScore)

	print("mean score cross val train :" , cross_valScore_train.mean())
	#/***param_grid ***/ 
	#### whats my grid parmas? 
	#print("para: " ,  linear_regression.get_params().keys())
	#para:  dict_keys(['copy_X', 'fit_intercept', 'n_jobs', 'normalize'])

	param_grid = [{'copy_X' : [0.0075,0.01, 0.075 ,0.15 , 0.25,0.5, 1 ,10] , 'fit_intercept' : [0.0075,0.01,0.075 ,0.15, 2, 4,6,8], 'n_jobs' : [0.0075,0.01,0.075 ,0.15, 0.5,1,2,4], 'normalize' : [0.0075,0.01,0.075 ,0.15, 0.5,1,2,4,6] }]
	print("Running gridSearchCV ... ")
	grid_search = GridSearchCV(linear_regression, param_grid, cv=10,scoring='neg_mean_squared_error', refit= True, n_jobs=-1)
	#fit it 
	grid_search.fit(predictors_processed, labels_processed)
	#best param
	score_gridCV = grid_search.best_score_ 
	best_params_gridCV = grid_search.best_params_
	best_est_gridCV = grid_search.best_estimator_
	#cv_results = grid_search.cv_results_
	print("Best GridSearchCV Params : ", best_params )
	print("Best GridSearchCV Estimator :" , best_est )
	print("Best GridSearchCV Score  " , score )
	#print("results : " , cv_results )
	# Test this estimator with our test data. 
	final__gridCV_model_est = grid_search.best_estimator_
    # first try it in corss val score
    valScore_est_grid = cross_val_score( final__gridCV_model_est,  predictors_processed ,labels_processed,cv=10, scoring = 'neg_mean_squared_error')
	print("mean socre of GridSearchCV estimator using cross val :  " , valScore_est_grid.mean())
	# use the estimator against our fit test data ! 
	predictor_pipeline_test = make_predictor_pipeline( )
	label_pipeline_test = make_label_pipeline( )
	predictors_processed_test = predictor_pipeline_test.fit_transform( test_data )
	labels_processed_test = label_pipeline_test.fit_transform( test_data )
	
	final_predictions = final__gridCV_model_est.predict(predictors_processed_test)
	mean_squared_error_test = sklearn.metrics.mean_squared_error( labels_processed_test, final_predictions )
	root_mean_squared_error_test = np.sqrt( mean_squared_error_test )
	print( "Error Test Data Using GridSearchCV: " + str( root_mean_squared_error_test ) )

	### DO THIS ALL OVER AGAIN WITH RANDMOIZED randmizedSearchCV
	#because why not
	print("Running RandomizedSearchCV...")
	n_iter_search = 20
	random_search = RandomizedSearchCV(linear_regression, param_distributions=param_grid,n_iter=n_iter_search, cv=10)
	start = time()
	#Fit it 
	random_search.fit(predictors_processed, labels_processed)
	#results ! 
	score_rand_search = random_search.best_score_ 
	best_params_rand_search = random_search.best_params_
	best_est_rand_search = random_search.best_estimator_
	#cv_results = random_search.cv_results_
	print("Best RandomizedSearchCV Params : ", best_params_rand_search )
	print("Best RandomizedSearchCV Estimator :" , best_est_rand_search )
	print("Best RandomizedSearchCV score: " , score_rand_search )

	final__randCV_model_est = grid_search.best_estimator_
    # first try it in corss val score
    valScore_est_rand = cross_val_score( final__randCV_model_est,  predictors_processed ,labels_processed,cv=10, scoring = 'neg_mean_squared_error')
	print("mean socre of RandomizedSearchCV estimator using cross val  :  " , valScore_est_rand.mean())
	# use the randmoized estimator against fit test data
	final_predictions01 = final__randCV_model_est.predict(predictors_processed_test)
	mean_squared_error_test01 = sklearn.metrics.mean_squared_error( labels_processed_test, final_predictions01 )
	root_mean_squared_error_test01 = np.sqrt( mean_squared_error_test01 )
	print( "Error Test Data Using RandomizedSearchCV: " + str( root_mean_squared_error_test01 ) )


	return


if __name__ == "__main__":
    main( )
