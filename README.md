# NanoGraphene Builder
The implemented class allows to build, plot, and save Nanographene structures which can then be used for further Quantum Chemistry simulations. 
Thus, the resulting geometries are not optimized and just follow from geometrical reasoning. Additionally, unpaired electrons are paired with hydrogen atoms. 
Each structure that is created is only created once during the respective run. Since it is based on a random walk approach, the created structures may vary between different runs. 

# Scalliing 
Currently, no parallelization is employed and the code was not designed to create structures with more than 10 Benzol rings. The most costly part currently is, 
to check if the resulting nanographene has already been created during this run. This needs to take rotational and translated versions of each one into account. 

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
In it, the geometry xyz files will be stored. The plots of the rings will be stored in the subfolder draws. 
