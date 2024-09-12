"""
Steps:
0. create a logging mechanism [DONE]
1. go through config tracker and all configurations passed and ask user if they
   wish to add new configs (have a flag to directly say yes)
2. update config tracker with new configurations to avoid deadlocks/race-conditions
3. check for roi_name and geometry conflicts and raise error if any present
4. create a copy of the datacube catalog with new name for each thread
5. create a list of cli inputs
6. execute cli inputs via multi-threading -- stagger the calls by random or fixed delays
7. merge the datacube catalogs in a non-conflicting manner
"""
