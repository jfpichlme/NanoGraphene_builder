# NanoGraphene Builder
The implemented class allows to build, plot, and save Nanographene structures which can then be used for further Quantum Chemistry simulations. 
Thus, the resulting geometries are not optimized and just follow from geometrical reasoning. Additionally, unpaired electrons are paired with hydrogen atoms. 
Each structure that is created is only created once during the respective run. Since it is based on a random walk approach, the created structures may vary between different runs. 

# Scalling 
Currently, no parallelization is employed. However, if the grid range is kept at medium sizes and the disk space allows to save enough structures including plots, the code can create a few thousand nanographene structures in a minute on a standard cpu. 

# Usage
The code requires the following packages: 
                                          - NumPy 
                                          - matplotlib 
                                          - networkx 
                                          - sklearn

# Input
In the main file, the number of benzol rings is defined when the class object is initialized. 

# Output 
When running, the script will automatically create a new folder with the current date and time as the name. 
In it, the geometry xyz and in files will be stored. The plots of the rings will be stored in the subfolder draws.
Additionally, the structures are stored in two different directories, open shell and closed shell. 

# Warning
There is no guarantee for completeness. This is just a sampling approach for quickly creating a large number of nanographene structures with no intend 
to include all for the specific amount of carbon atoms. 
