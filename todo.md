

utils_data:
-------------------------------------------------
data_interval(low_b,high_b,N=100)
rename with random_values(low_b,high_b,size)

data(N = 10)
rename with random_points(size)

data_from_file(filepath)
rename dataset_from_file

build_training_set(data, func)
rename build_dataset

build_training_set_fmultipleparams(data,func)
rename build_dataset_fmultiparams

target_vector(t_set)
rename target_vector(dataset)

input_data_matrix(t_set)
rename data_matrix(dataset)

utils_math:
------------------------------------------------
randomline()
rename random_line_coefs

target_function(coords)
rename linear_function(coefs)

target_random_function(coords)
rename random_function()...

signex(x,compare_to = 0)
delete use sign with an alias

sign(x,compare_to = 0)
map_point(point,f)
rename get_sign(point,f)

map_point_fmultipleparams(point,f)
get_sign_multiparam

pseudo_inverse(X)
check if needed

linear_regression(N_points,t_set,lda = 1.0)

linear_regression_lda(N_points,t_set,lda)

utils_print:
-----------------------------------------------
print_avg(name,vector)
