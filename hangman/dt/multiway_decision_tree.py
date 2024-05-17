'''
	Package: cs771
	Module: multiway_decision_tree
	Author: puru
	Institution: CSE, IIT Kanpur
	License: MIT License
	
	Give skeletal support for implementing a multiway decision tree
'''

import numpy as np
import warnings
import sys

class Node:
	# A node stores its own depth (root = depth 0), a link to its parent and an actor that
	# splits data points for internal nodes and makes final predictions for leaf nodes
	# A dictionary is used to store the children of a non-leaf node. Each child is paired
	# with the corresponding outcome. A node also stores the path that led to it
	def __init__( self, depth, parent, size ):
		self.depth = depth
		self.actor = None
		self.parent = parent
		self.children = {}
		self.is_leaf = True
		self.ancestor_splits = []
		self.size = size
	
	# get_split_actor should take training points and the ancestor splits done till this point
	# and return a split actor object capable of splitting data points among children of this node.
	# get_leaf_actor should return a leaf actor object that does the final prediction
	# The objects returned by the get_split_actor and get_leaf_actor methods should internally store
	# any data they require to do their job properly
	def train( self, trn_pts, get_split_actor, get_leaf_actor, min_leaf_size, max_depth, is_pure_enough, get_size, fmt_str, verbose ):
		# If this node has too few data points, or is too deep in the tree, or is too pure, make it a leaf
		if self.size <= min_leaf_size or self.depth >= max_depth or is_pure_enough( trn_pts ):
			self.is_leaf = True
			self.actor = get_leaf_actor( trn_pts, self.ancestor_splits )
			if verbose:
				print( '█' )
		else:
			# This node will be split and hence it is not a leaf
			self.is_leaf = False
			
			# Get the best possible decision stump
			# Get a dictionary mapping possible outcomes to the training points with that outcome
			# Note that unique_outcomes and trn_splits must be lists of the same size
			( self.actor, split_dict ) = get_split_actor( trn_pts, self.ancestor_splits )
			
			# Get the attribute(s) on which this stump split the data
			split_attr = self.actor.get_attr()
			
			if verbose:
				print( split_attr )
			
			# Create a new child node for each possible outcome
			for ( i, ( outcome, trn_split ) ) in enumerate( split_dict.items() ):
				
				if i == len( split_dict ) - 1:
					if verbose:
						print( fmt_str + "└───", end = '' )
					next_fmt_str = fmt_str + "    "
				else:
					if verbose:
						print( fmt_str + "├───", end = '' )
					next_fmt_str = fmt_str + "│   "
				
				self.children[ outcome ] = Node( depth = self.depth + 1, parent = self, size = get_size( trn_split ) )
				self.children[ outcome ].ancestor_splits = self.ancestor_splits.copy()
				self.children[ outcome ].ancestor_splits.append( [ split_attr, outcome ] )
				
				# Recursively train this child node
				self.children[ outcome ].train( trn_split, get_split_actor, get_leaf_actor, min_leaf_size, max_depth, is_pure_enough, get_size, next_fmt_str, verbose )
	
	# The split method of a split actor should return a dictionary of outcomes as keys and
	# the set of corresponding test points that had those outcomes and their original indices
	# Note that the split method may also update the features of the data point as a side effect
	# The predict method of a leaf actor should return a list of predictions
	def predict( self, tst_pts ):
		if self.is_leaf:									# If I am a leaf I can predict rightaway
			return self.actor.predict( tst_pts, self.ancestor_splits )
		else:												# Else I have to ask one of my children to do the job
			split_dict = self.actor.split( tst_pts, self.ancestor_splits )
			
			pred = []
			indices = []
			
			for ( outcome, ( idx, tst_split ) ) in split_dict.items():
				
				# This should ideally not happen -- improve the stump implementation if this is happening
				# The stump should ensure that all possibilities are covered, e.g. by having a catch-all outcome
				# The predict and default_predict methods must always return lists, even if they are singleton
				if outcome not in self.children:
					warnings.warn( "Unseen outcome " + str( outcome ) + " -- using the default_predict routine", UserWarning )
					child_pred = self.actor.default_predict( tst_split, self.ancestor_splits )
				else:
					child_pred = self.children[ outcome ].predict( tst_split )
				
				pred += child_pred
				indices += idx
			
			idx_sort = np.argsort( indices )
			
			ord_pred = [ pred[ idx_sort[ i ] ] for i in np.arange( len( indices ) ) ]
			
		return ord_pred

class Tree:
	def __init__( self, min_leaf_size = 1, max_depth = 6 ):
		self.min_leaf_size = min_leaf_size
		self.max_depth = max_depth
	
	def predict( self, tst_pts ):
		return self.root.predict( tst_pts )
	
	def train( self, trn_pts, get_split_actor, get_leaf_actor, is_pure_enough, get_size, verbose = False ):
		self.root = Node( depth = 0, parent = None, size = get_size( trn_pts ) )
		if verbose:
			print( "root" )
			print( "└───", end = '' )
		self.root.train( trn_pts, get_split_actor, get_leaf_actor, self.min_leaf_size, self.max_depth, is_pure_enough, get_size, "    ", verbose )
