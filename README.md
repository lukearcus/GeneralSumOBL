# General Sum Off-Belief Learning

run main.py for rl, obl or ot-rl. Run main\_FSP.py for fictitious self play, the following options can be used for either (although some will not have any effect on FSP).

## options ##

	**--lvls** LEVELS

		Select number of OBL/OT_RL levels to run through, defaults to 10.

	**--game** kuhn/leduc

		Choose either kuhn poker or leduc hold 'em.

	**-ab, --avg_bel**

		Generate an averaged belief (over levels), and use this in OBL. 

	**-ap, --avg_pol**

		Generate the averaged policy across levels and use this when evaluating.

	**-al, --avg_learn**

		When carrying out OBL, use the opponent's averaged policy to find their action.

	**-a, --all_avg**

		Averaged belief, policy and learning.

	**--debug**

		Prints out debugging information.

	**-v**

		Prints out some information about progress.

	**--learner** LEARNER_CHOICE
		
		Choose learner from rl, ot_rl or obl, for learning uising obl. Defaults to obl	

	**--fsp**

		Uses FSP to learn nash, obl is default.

	**--obl**
		Uses obl/rl/ot_rl.

Example usage:
	python main.py -a -v --obl --lvls 5

## Dependencies ##


### Name

matplotlib	

numpy

scipy
