How to use this branch:

1. merge the branch you want to benchmark with this branch (locally, do not push to this branch)
2. you can run the benchmarks by running the file 'run_benchmarks.py'
   - you can set some parameters at the top of the file
   - changing the branch name parameter will make it easier to read the plot results.
3. to look at the results, run 'results.py'
   - The total times for each operation, for each csv file will be plotted together
   - you can set a threshold, and a baseline file, to see if any instances/transformations became a lot slower/faster
   - I added a baseline file, but that was locally run on my machine, so it might be better to generate your own.