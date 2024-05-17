import random

class Merlin:
	def __init__( self, err_max ):
		self.secret = ""
		self.reveal = ""
		self.err_max = err_max
		self.tot_err_count = 0
		self.rnd_err_count = 0
		self.fail_count = 0
		self.fail_list = []
		self.win_list = []
		self.arthur = None
		
	def meet( self, arthur ):
		self.arthur = arthur
	
	def reset( self, secret ):
		self.secret = secret
		self.reveal = []
		self.rnd_err_count = 0
		
		for c in self.secret:
			self.reveal.append( '_' )
	
	# Receive a message from Arthur
	# Work on it and message Arthur back
	def msg( self, guess, done = False ):
	
		# Guessing 0 is a way for Arthur to give up
		# This option should ideally never be used
		if guess == '0':
			print( "Warning: Arthur has given up on the word", self.secret )
			self.tot_err_count += self.rnd_err_count
			self.fail_count += 1
			return
		
		# Find out the correctness of the guess(es)
		for c in guess:
			error = True
			
			for i in range( len( self.secret ) ):
				if self.secret[i] == c:
					self.reveal[i] = c
					error = False
			
			if error:
				self.rnd_err_count += 1
		
		# Too many mistakes were made - terminate the round
		if self.rnd_err_count >= self.err_max:
			self.fail_count += 1
			self.fail_list.append( self.secret )
			self.tot_err_count += self.rnd_err_count
			return
		
		# Mistakes were under threshold and the word was correctly guessed
		if '_' not in self.reveal:
			self.win_list.append( self.secret )
			self.tot_err_count += self.rnd_err_count
			return
		
		# Arthur has given up and neither succeded nor exceeded the error threshold
		if done:
			self.fail_count += 1
			self.fail_list.append( self.secret )
			self.tot_err_count += self.rnd_err_count
			return
		
		# If none of the above happen, continue playing
		self.arthur.msg( ' '.join( self.reveal ) )
	
	def reset_and_play( self, secret ):
		self.reset( secret )
		self.arthur.msg( ' '.join( self.reveal ) )
