# Motivation

Allow for maximum parallelism in execution of sorting algorithms to make use of the fact tha computing power improves rapidly in its thread-count, but not single-thread performance. Potentially improve average time complexity of algorithms.

# Architecture

The architecture of the system.

## General idea

Since ANNs are function-approximators, a precise sorting of number using only one of them cannot be expected. Instead, an ANN followed by a regular sorting algorithm might be a good alternative:

- The ANN would pre-sort the numbers in a way that is good for the sorting algorithm (some sorting algorithms have a time-complexity for some orderings that is much lower than the worst-case one.)
- The ANN might be slightly inefficient, but if it reduces the average time complexity of an algorithm, it would almost certainly be worth it for large lists, and it might be worth it even if this isn't the case, as long as it can increase processor-usage. In any case, it's an interesting project.

## Plan

- The network architecture
	- A transformer seems reasonable (for seeing wide-reaching connections / having a birds-eye view)
		- Therefore, read paper.
	- General:
		- Input: fixed-sized list of positive integers (for now)
		- Output: equally large list of real numbers indicating the index that the corresponding input-number should belong to.
	- Make output into a list -> give list to sorting algorithm
		- Learn about sorting algorithms: which ones benefit from plausibly achievable pre-sorting? What are their performance characteristics?
	- Activation function:
		- Appy `ReLU`: allows neural network to just assign negative numbers to small inputs for easy pre-sorting.
		- Apply `torch.where(x > list_len, list_len - 1, x)`: allows layer to assign a very large index to large numbers for easy pre-sorting.
		- Both of the above might be especially useful for a transformer which sees the whole list and might just push large numbers in one direction and small ones in the other.
- Turning approximate indices into actual, usable ones
	- This might require sorting which would make the whole exercise infinitely recursive.
- The loss function. 
	- Several possibilities:
		- Distance of output from correct ordering -> endorse closeness to acutal index.
		- Track number of times that the sorting algorithm has to touch each index in order to get it correctly sorted 
			- Maybe use this number directly as a loss for each index to minimize the number of orderings
			- Or use the total count: in some cases, it might be beneficial to pre-sort some number in such a way as to have them be moved often afterwards, if it means that the other numbers have to be moved rarely. This shouldn't be discouraged too strongly. 
			- A combination of both might be used (but too many losses aren't good, either) -> think about this some more.
	- Use some combination of them.
- End-to-end training
	- Two possibilities:
		1. A simple `Sequential` multi-layered network that changes the numbers into  indices
		2. Many single-layered nets that each see the actual list that is to be sorted. 
	- To elaborate on number 2:
		- Create an `Index`-class which inherits from `Module`.
			- `forward`: transforms the approximate indices of the previous layer into precice indices and outputs the corresponding list as an input to the next layer.
				
				- 
			- `backward`: Takes gradient of known length and reorders it in the reverse direction of the forward reordering of the list.