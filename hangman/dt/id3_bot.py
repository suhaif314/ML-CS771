import pickle

class Split_actor:
	def __init__( self, attr, outcomes ):
		self.attr = attr

class Leaf_actor:
	def __init__( self, data ):
		self.data = data

class Arthur:
	def __init__( self ):
		
		with open( "dt.mdl", 'rb' ) as f:
			self.dt = pickle.load( f )
			
		self.node_to_ask = self.dt.root
		self.merlin = None
		self.is_done = False
		
		
	def meet( self, merlin ):
		self.merlin = merlin
	
	def reset( self ):
		self.node_to_ask = self.dt.root
		self.is_done = False
	
	def msg( self, mask ):
	
		# Are we at a leaf?
		if self.node_to_ask.is_leaf:
			
			# I finished the round but for some reason
			# Merlin is still asking me questions
			# Something is wrong - better give up!
			if self.is_done:
				print( "I am already done!" )
				self.merlin.msg( '0' )
			else:
				# Get of all guesses that have been made so far
				ancestor_attr = [ split[0] for split in self.node_to_ask.ancestor_splits ][1:]
				word = self.node_to_ask.actor.data[0]
				guess = []
				
				# Collect all unguessed characters in the word and guess them in one go
				for c in word:
					if c not in ancestor_attr:
						guess.append( c )
				
				# I am done playing this round
				self.is_done = True
				self.merlin.msg( "".join( guess ), done = True )
		else:
			# Find the child that handles this mask
			self.node_to_ask = self.node_to_ask.children[ mask ]
			
			# If that child is a leaf, can directly predict
			# Since leaves do not ask any questions
			if self.node_to_ask.is_leaf:
				self.msg( mask )
			else:	# Send Merlin the question that child would have asked
				self.merlin.msg( self.node_to_ask.actor.attr )
