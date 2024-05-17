import predict
import time as tm
import numpy as np

num_test = 7
filepaths = [ "test/%d.png" % i for i in range( num_test ) ]
file = open( "test/labels.txt", "r" )
gold_output = file.read().splitlines()
file.close()

# Get recommendations from predict.py and time the thing
tic = tm.perf_counter()
output = predict.decaptcha( filepaths )
toc = tm.perf_counter()

parity_match = np.array( [ 1 if x.strip().upper() == y.strip().upper() else 0 for ( x, y ) in zip ( output, gold_output ) ] ).sum()

print( f"Time taken per image is {(toc - tic) / num_test} seconds" )
print( f"Parity match score is {parity_match / num_test}" )
