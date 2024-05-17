from id3_bot import Arthur
from id3_bot import Split_actor
from id3_bot import Leaf_actor
from hangman_bot import Merlin
import time
import colorama

colorama.init( convert = True )

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

merlin = Merlin( err_max )
arthur = Arthur()
merlin.meet( arthur )
arthur.meet( merlin )

f = open( "base", 'r' )
words = f.read().split( '\n' )[:-1]		# Omit the last line since it is empty
f.close()
num_words = len( words )

for (i, secret) in enumerate( words ):
	arthur.reset()
	merlin.reset_and_play( secret )
	completion = round( i * 75 / num_words )
	padding = ( 20 - len( secret ) ) * ' '
	if completion < 25:
		color = colorama.Fore.LIGHTRED_EX
	elif completion < 50:
		color = colorama.Fore.LIGHTYELLOW_EX
	else:
		color = colorama.Fore.LIGHTGREEN_EX
	print( color + f"\r│{ completion * '█' }{ ( 75 - completion ) * '.' }│ ({ i + 1 }/{ num_words }) { secret }{ padding }", end = '', flush = True )
	if i % 20 == 0:
		time.sleep( 1e-3 )

print( colorama.Fore.RESET )
print( f"\n{ num_words } words, { merlin.fail_count } losses, and total { merlin.tot_err_count } errors (avg { merlin.tot_err_count / num_words } errors per word )" )
usr_in = input( "Press 'w' for win list, 'l' for loss list, 'wl' for both lists, and any other character for neither: " )
if 'w' in usr_in:
	print( f"\n\n\nSuccess cases: { merlin.win_list }" )
if 'l' in usr_in:
	print( f"\n\n\nFailure cases: { merlin.fail_list }" )
