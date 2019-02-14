import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.metrics
from sklearn.externals import joblib



#open data file
def get_data( filename ):
	print("given filename: ", filename)
	#we have added index_col to none to not ignore any col unlike first assignment
	data = pd.read_csv( filename, index_col=None )
	#print(data.columns.values)
	return data

#display data 
def display_data( data ):
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


def split_data( data, ratio ):
    data_train, data_test = sklearn.model_selection.train_test_split( data, test_size=ratio )
    data_train.to_csv( "mo_train.csv" )
    data_test.to_csv( "mo_test.csv" )
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




def main( ):
	print("Reading data...")
	#numpy show
	raw_data = get_data("data.csv")

	display_data(raw_data)
	# #printing scalers to know the train_x values to find correlations
	# data_train = get_data( "mo_train.csv" )
	# #display_data("data_train")
	# train_X_raw, train_Y = separate_predictors_and_labels( data_train )
	# train_X, scaler = scale_predictors( train_X_raw )
	# print ("Train_X_RAW")
	# print( train_X_raw )
	# print ("train_X")
	# print( train_X )
	# print ("sclaer")
	# print( scaler )
	# reg = fit( train_X, train_Y )
	# #train_X.to_csv("train_X.csv")
	# #print(data.Concrete_compressive_strength)
	# data_test = get_data( "mo_test.csv" )
	# test_X_raw, test_Y = separate_predictors_and_labels( data_test )
	# test_X_raw = test_X_raw.astype( 'float64' )
	# test_X = scaler.transform( test_X_raw )
	# test( test_X, test_Y, reg )
	# #Dump data into joblib 
	# save_to_joblib(reg)

	return


if __name__ == "__main__":
    main( )
