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
- The loss function. 
	- Several possibilities:
		- Distance of output from correct ordering -> endorse closeness to acutal index.
		- Track number of times that the sorting algorithm has to touch each index in order to get it correctly sorted 
			- Maybe use this number directly as a loss for each index to minimize the number of orderings
			- Or use the total count: in some cases, it might be beneficial to pre-sort some number in such a way as to have them be moved often afterwards, if it means that the other numbers have to be moved rarely. This shouldn't be discouraged too strongly. 
			- A combination of both might be used (but too many losses aren't good, either) -> think about this some more.
	- Use some combination of them.