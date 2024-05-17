import random

if not __name__ == "__main__":
	print( "This code must be invoked directly." )
	exit()

usr_in = input( "Enter the maximum number of mistakes (blank input => 6): " )
while True:
	if len( usr_in ) == 0:
		err_max = 6
	else:
		try:
			err_max = int( usr_in )
		except:
			usr_in = input( "Invalid input ... try again (blank input => 6): " )
			continue
	break

f = open( "base", 'r' )
words = f.read().split( '\n' )[:-1]		# Omit the last line since it is empty
f.close()

while True:
	secret = words[ random.randint( 0, len(words) - 1 ) ]
	
	# secret = "academic"
	# secret = simultaneously
	
	reveal = []
	for c in secret:
		reveal.append( '_' )
	
	err_count = 0
	guess_count = 0
	
	while err_count < err_max and '_' in reveal:
		print( "\n\n\nHere is your word: ", ' '.join( reveal ) )
		print( "Mistake bar: " + "│" + "█" * err_count + " " * (err_max - err_count) + "│ (" + str( err_count ) + "/" + str( err_max ) + ")" )
		usr_in = input( "Enter your character guess(es): " )
		
		while True:
			try:
				guess = str( usr_in ).lower()
				if not guess.isalpha():
					usr_in = input( "Invalid input ... try again: " )
					continue
			except:
				usr_in = input( "Invalid input ... try again: " )
				continue
			break
		
		guess_count += len( guess )
		
		for c in guess:
			error = True
			
			for i in range( len( secret ) ):
				if secret[i] == c:
					reveal[i] = c
					error = False
			
			if error:
				err_count += 1
	
	if err_count >= err_max: 
		print( "\n\n\nMistake bar: " + "│" + "█" * err_count + " " * (err_max - err_count) + "│ (" + str( err_count ) + "/" + str( err_max ) + ")" )
		print( "Sorry! you have reached the mistake threshold" )
		print( "The word was: \"", secret, "\"" )
	else:
		print( "\n\n\nGreat ... you correctly guessed the word: \"", secret, "\""  )
		print( "You took", guess_count, "guesses and made", err_count, "mistakes" )
	
	usr_in = input( "\n\n\nEnter 'Y' or 'y' to continue:" )
	if not str(usr_in).lower() == 'y':
		exit()