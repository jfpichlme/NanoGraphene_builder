import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import random
import os

import networkx as nx
from sklearn.neighbors import NearestNeighbors


class Creator:
    '''Class that randamly builds Nanographene structures. The user inserts the max number of rings that should be
        included in the structures'''

    def __init__(self, nr_rings, max_number_conn, hydrogenise=True, grid_size=20):

        self.nr_rings = nr_rings
        self.hydrogenise = hydrogenise
        self.grid_size = grid_size
        self.nr_connections = nr_rings - 1
        now = datetime.now()
        self.fold_date = str(now)
        self.fold_date = self.fold_date.replace(':', '-')
        self.max_number_conon = max_number_conn

    def set_up_grid(self):
        # simple function that sets up a 2D grid without coordinate filling
        # Create a 2D grid from scratch using a list of lists and completely fill it with data.
        dict_grid = {}
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                dict_grid[(x, y)] = np.asarray(
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])

        # create a grid object containing all the lattice points
        self.grid_object = dict_grid

    def create_graphene_structure(self):
        # function that fills array with graphene coordinates
        self.grid_object[(0, 0)] = np.asarray(
            [[8.739, 7.5, 7.5], [9.979, 8.215, 7.5], [9.979, 9.647, 7.5], [8.739, 10.362, 7.5],
             [7.5, 9.647, 7.5], [7.5, 8.215, 7.5]])

        # checks if we have the shift between the benzol rings
        parity_flag = 1

        # dict diff contains the distance to the neighbouring structures
        dict_diff = {}

        # initialize the first element
        dict_diff[(0, 0)] = [1.1, 1.1, 1.0, 1.0, 1.8, 1.8, 1.1, 1.1]

        # loop over all entries in the dictionary and assign correct carbon atoms
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if x == 0:
                    continue
                else:
                    self.grid_object[(x, y)] = self.grid_object[(x - 1, y)] + np.asarray(
                        [[2.48, 0, 0], [2.48, 0, 0], [2.48, 0, 0],
                         [2.48, 0, 0], [2.48, 0, 0], [2.48, 0, 0]])

                    dict_diff[(x, y)] = dict_diff[(x - 1, y)]

            # reset the x coordinate to the very left side to introduce shift in graphene like structure
            x = 0
            if parity_flag < 1:
                parity_flag = 1

                # define one element on the very left to have one difference configuration
                dict_diff[(x, y + 1)] = [1.1, 1.1, 1.0, 1.0, 1.8, 1.8, 1.1, 1.1]

            else:
                parity_flag = -1

                # define one element on the very left to have one difference configuration
                dict_diff[(x, y + 1)] = [1.1, 1.1, 1.0, 1.0, 1.1, 1.1, 1.8, 1.8]

            x_shift = parity_flag * 1.24
            # introduce shift to very left entries
            self.grid_object[(x, y + 1)] = self.grid_object[(x, y)] + np.asarray(
                [[x_shift, 2.147, 0], [x_shift, 2.147, 0], [x_shift, 2.147, 0],
                 [x_shift, 2.147, 0], [x_shift, 2.147, 0], [x_shift, 2.147, 0]])

        # add dictinoary containing the graphene centere differences to the class
        self.dict_diff = dict_diff

    def plot_grid(self):
        # simply loops over the entries in the dictionary and does a scatter plot

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for coord in self.grid_object[(x, y)]:
                    plt.scatter(coord[0], coord[1], s=2)

        plt.show()

    @staticmethod
    def perform_random_walk_straight(nr_conn):
        # performs n^2 random walks where n is the number of connections = number of benzol rings + 1

        # define collections of points of random walk
        structs = []

        # counter of how many are rejected
        count = 0

        # define start position of random walk, always start at the same center point
        start = [10, 10]

        # array contianing accepted differences
        acc_diff = []

        # array containing the differences from the starting position
        diff_array = []

        for trial in range(nr_conn ** 5):

            # helper array that contains the walks that are stored
            trial_walk = []

            # array containing the differences from the starting position
            diff_array = []

            # initialize start of walk
            trial_walk.append(start)

            # initialize count variable
            step = 0

            while step < nr_conn:

                # first we query if we should go in x or y direction
                random_nr = random.random()
                if random_nr <= 0.5:
                    # go x
                    if random.random() <= 0.5:
                        # go left
                        pot_step = [trial_walk[-1][0] - 1, trial_walk[-1][1]]
                        # pot_diff = self.dict_diff[(pot_step[0], pot_step[1])][3]

                        # pot_diff = np.around(np.linalg.norm(np.asarray(pot_step) - np.asarray(start)), 1)
                        if pot_step in trial_walk:
                            continue
                        else:
                            trial_walk.append(pot_step)
                            # diff_array.append(pot_diff)
                            step = step + 1
                    else:
                        # go right
                        pot_step = [trial_walk[-1][0] + 1, trial_walk[-1][1]]
                        # pot_diff = self.dict_diff[(pot_step[0], pot_step[1])][2]

                        # pot_diff = np.around(np.linalg.norm(np.asarray(pot_step) - np.asarray(start)), 1)
                        if pot_step in trial_walk:
                            continue
                        else:
                            trial_walk.append(pot_step)
                            # diff_array.append(pot_diff)
                            step = step + 1
                else:
                    # go y
                    if random.random() <= 0.5:
                        # go up
                        pot_step = [trial_walk[-1][0], trial_walk[-1][1] + 1]
                        # pot_diff = self.dict_diff[(pot_step[0], pot_step[1])][0]

                        # pot_diff = np.around(np.linalg.norm(np.asarray(pot_step) - np.asarray(start)), 1)
                        if pot_step in trial_walk:
                            continue
                        else:
                            trial_walk.append(pot_step)
                            # diff_array.append(pot_diff)
                            step = step + 1
                    else:
                        # go down
                        pot_step = [trial_walk[-1][0], trial_walk[-1][1] - 1]
                        # pot_diff = self.dict_diff[(pot_step[0], pot_step[1])][1]

                        # pot_diff = np.around(np.linalg.norm(np.asarray(pot_step) - np.asarray(start)), 1)
                        if pot_step in trial_walk:
                            continue
                        else:
                            trial_walk.append(pot_step)
                            # diff_array.append(pot_diff)
                            step = step + 1

            # first we loop over all positions that are contained in the path and adapt them according to the following
            # coordinate change :

            # .   .   .   .   .   .                     # .   .   .   .   .   .
            # .   .   .   .   .   .                     #   .   .   .   .   .   .
            # .   .   .   .   .   .     -------->       # .   .   .   .   .   .
            # .   .   .   .   .   .                     #   .   .   .   .   .   .
            # .   .   .   .   .   .                     # .   .   .   .   .   .

            # For this the y values are multiplied by a factor of sqrt(3)/2 and each second x value is shifted by one half
            coords_shifted = []

            for coord in trial_walk:

                # first we calculate the difference to the start vector to check if we shift the x values
                diff_vec_start = np.abs(np.asarray(coord) - np.asarray(start))

                # then we adapt the coordinates accordingly
                if diff_vec_start[1] % 2 == 0:
                    coords_shifted.append([coord[0] + 0.5, coord[1] * np.sqrt(3) / 2])

                else:
                    coords_shifted.append([coord[0], coord[1] * np.sqrt(3) / 2])

            # next we calculate the differences between all points
            for vec1 in coords_shifted:

                for vec2 in coords_shifted:
                    # first we calculate the difference vector
                    diff_vec_abs = np.abs(np.asarray(vec2) - np.asarray(vec1))

                    # next we calculate the total difference
                    diff_array.append(np.round(np.sqrt(np.sqrt(diff_vec_abs[0] ** 2 + (diff_vec_abs[1] ** 2))), 2))

            # sort array for checking if it exists
            diff_array.sort()

            exists = diff_array in acc_diff

            if exists == False:
                structs.append(trial_walk)
                acc_diff.append(diff_array)
            else:
                count = count + 1

        print("Fraction that got declined: ", count / (nr_conn ** 5))

        return structs

    @staticmethod
    def perform_random_walk_all(nr_conn):
        # performs n^2 random walks where n is the number of connections = number of benzol rings + 1

        # define collections of points of random walk
        structs = []

        # counter of how many are rejected
        count = 0

        # define start position of random walk, always start at the same center point
        start = [10, 10]

        # array contianing accepted differences
        acc_diff = []

        # array containing the differences from the starting position
        diff_array = []

        for trial in range(nr_conn ** 5):

            # helper array that contains the walks that are stored
            trial_walk = []

            # array containing the differences from the starting position
            diff_array = []

            # initialize start of walk
            trial_walk.append(start)

            # initialize count variable
            step = 0

            while step < nr_conn:

                # first we query if we should go in x or y direction
                random_nr = random.random()
                if random_nr <= 0.3:
                    # go x
                    if random.random() <= 0.5:
                        # go left
                        pot_step = [trial_walk[-1][0] - 1, trial_walk[-1][1]]
                        # pot_diff = self.dict_diff[(pot_step[0], pot_step[1])][3]

                        # pot_diff = np.around(np.linalg.norm(np.asarray(pot_step) - np.asarray(start)), 1)
                        if pot_step in trial_walk:
                            continue
                        else:
                            trial_walk.append(pot_step)
                            # diff_array.append(pot_diff)
                            step = step + 1
                    else:
                        # go right
                        pot_step = [trial_walk[-1][0] + 1, trial_walk[-1][1]]
                        # pot_diff = self.dict_diff[(pot_step[0], pot_step[1])][2]

                        # pot_diff = np.around(np.linalg.norm(np.asarray(pot_step) - np.asarray(start)), 1)
                        if pot_step in trial_walk:
                            continue
                        else:
                            trial_walk.append(pot_step)
                            # diff_array.append(pot_diff)
                            step = step + 1
                elif random_nr <= 0.6:
                    # go y
                    if random.random() <= 0.5:
                        # go up
                        pot_step = [trial_walk[-1][0], trial_walk[-1][1] + 1]
                        # pot_diff = self.dict_diff[(pot_step[0], pot_step[1])][0]

                        # pot_diff = np.around(np.linalg.norm(np.asarray(pot_step) - np.asarray(start)), 1)
                        if pot_step in trial_walk:
                            continue
                        else:
                            trial_walk.append(pot_step)
                            # diff_array.append(pot_diff)
                            step = step + 1
                    else:
                        # go down
                        pot_step = [trial_walk[-1][0], trial_walk[-1][1] - 1]
                        # pot_diff = self.dict_diff[(pot_step[0], pot_step[1])][1]

                        # pot_diff = np.around(np.linalg.norm(np.asarray(pot_step) - np.asarray(start)), 1)
                        if pot_step in trial_walk:
                            continue
                        else:
                            trial_walk.append(pot_step)
                            # diff_array.append(pot_diff)
                            step = step + 1

                else:

                    random_nr_diag = random.random()
                    # go y
                    if random_nr_diag <= 0.25:
                        # go down-right
                        pot_step = [trial_walk[-1][0] + 1, trial_walk[-1][1] - 1]
                        # pot_diff = self.dict_diff[(pot_step[0], pot_step[1])][5]

                        # pot_diff = np.around(np.linalg.norm(np.asarray(pot_step) - np.asarray(start)), 1)
                        if pot_step in trial_walk:
                            continue
                        else:
                            trial_walk.append(pot_step)
                            # diff_array.append(pot_diff)
                            step = step + 1

                    elif random_nr_diag <= 0.50:
                        # go up-right
                        pot_step = [trial_walk[-1][0] + 1, trial_walk[-1][1] + 1]
                        # pot_diff = self.dict_diff[(pot_step[0], pot_step[1])][4]

                        # pot_diff = np.around(np.linalg.norm(np.asarray(pot_step) - np.asarray(start)), 1)
                        if pot_step in trial_walk:
                            continue
                        else:
                            trial_walk.append(pot_step)
                            # diff_array.append(pot_diff)
                            step = step + 1

                    elif random_nr_diag <= 0.75:
                        # go left-up
                        pot_step = [trial_walk[-1][0] - 1, trial_walk[-1][1] + 1]
                        # pot_diff = self.dict_diff[(pot_step[0], pot_step[1])][6]

                        # pot_diff = np.around(np.linalg.norm(np.asarray(pot_step) - np.asarray(start)), 1)
                        if pot_step in trial_walk:
                            continue
                        else:
                            trial_walk.append(pot_step)
                            # diff_array.append(pot_diff)
                            step = step + 1

                    else:
                        # go left-down
                        pot_step = [trial_walk[-1][0] - 1, trial_walk[-1][1] - 1]

                        # pot_diff = self.dict_diff[(pot_step[0], pot_step[1])][7]
                        if pot_step in trial_walk:
                            continue
                        else:
                            trial_walk.append(pot_step)
                            # diff_array.append(pot_diff)
                            step = step + 1

            # first we loop over all positions that are contained in the path and adapt them according to the following
            # coordinate change :

            # .   .   .   .   .   .                     # .   .   .   .   .   .
            # .   .   .   .   .   .                     #   .   .   .   .   .   .
            # .   .   .   .   .   .     -------->       # .   .   .   .   .   .
            # .   .   .   .   .   .                     #   .   .   .   .   .   .
            # .   .   .   .   .   .                     # .   .   .   .   .   .

            # For this the y values are multiplied by a factor of sqrt(3)/2 and each second x value is shifted by one half
            coords_shifted = []

            for coord in trial_walk:

                # first we calculate the difference to the start vector to check if we shift the x values
                diff_vec_start = np.abs(np.asarray(coord) - np.asarray(start))

                # then we adapt the coordinates accordingly
                if diff_vec_start[1] % 2 == 0:
                    coords_shifted.append([coord[0] + 0.5, coord[1] * np.sqrt(3) / 2])

                else:
                    coords_shifted.append([coord[0], coord[1] * np.sqrt(3) / 2])

            # next we calculate the differences between all points
            for vec1 in coords_shifted:

                for vec2 in coords_shifted:
                    # first we calculate the difference vector
                    diff_vec_abs = np.abs(np.asarray(vec2) - np.asarray(vec1))

                    # next we calculate the total difference
                    diff_array.append(np.round(np.sqrt(np.sqrt(diff_vec_abs[0] ** 2 + (diff_vec_abs[1] ** 2))), 2))

            # sort array for checking if it exists
            diff_array.sort()

            exists = diff_array in acc_diff

            if exists == False:
                structs.append(trial_walk)
                acc_diff.append(diff_array)
            else:
                count = count + 1

        print("Fraction that got declined: ", count / (nr_conn ** 5))

        return structs

    def get_unique_atom_position(self, struct):
        # methode that gets all atoms contained in the given structure and

        # array containing the coordinates of the atoms in structure
        atom_coords = []

        for benzol_id in struct:
            for atom in self.grid_object[(benzol_id[0], benzol_id[1])]:
                atom_coords.append(np.around(atom, 1))

        atom_coords = np.unique(np.asarray(atom_coords), axis=0)

        return atom_coords

    @staticmethod
    def get_unique_hydrogen_positions(carbon_positions):
        # function that ads a hydrogen atom to each carbon atom that does not have 3 nearest neighbours

        # distance of sqrt of hydrogen atoms to the neighbouring carbon atoms
        distance = 1.1

        # First we need to identify those carbon atoms that do have 3 nearest neighbours
        carbon_positions2D = np.delete(carbon_positions, -1, axis=1)

        # use nearest neighbour algorithm to get two nearest neighbours (3 because itself is the nearest)
        nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(carbon_positions2D)

        # list of hydrogen atoms
        hydro = []

        # get distances and indices
        distances, indices = nbrs.kneighbors(carbon_positions2D)

        # we loop over all indices and check if the 3rd nearest neighbour has a distance that is smaller than 2.0, then
        # the carbon atom has a bonding with three carbon atom, thus no free electron to bind a hydrogen
        for i in range(len(indices)):

            if distances[i][3] < 2.0:
                continue

            else:
                # at first we calculate the point which lies in the center between the two neighbouring

                delta = np.round(1 / 2 * (carbon_positions2D[indices[i][1]] - carbon_positions2D[indices[i][2]]), 1)

                P = carbon_positions2D[indices[i][2]] + delta

                # overall we have 6 conditions where the H atom should be added
                if delta[1] == 0:
                    if P[1] > carbon_positions2D[indices[i][0]][1]:
                        hydro.append([carbon_positions2D[indices[i][0]][0],
                                      carbon_positions2D[indices[i][0]][1] - distance, carbon_positions[i][-1]])

                    else:
                        hydro.append([carbon_positions2D[indices[i][0]][0],
                                      carbon_positions2D[indices[i][0]][1] + distance, carbon_positions[i][-1]])

                elif delta[0] == 0:
                    if P[0] > carbon_positions2D[indices[i][0]][0]:
                        hydro.append([carbon_positions2D[indices[i][0]][0] - distance,
                                      carbon_positions2D[indices[i][0]][1], carbon_positions[i][-1]])

                    else:
                        hydro.append([carbon_positions2D[indices[i][0]][0] + distance,
                                      carbon_positions2D[indices[i][0]][1], carbon_positions[i][-1]])


                # first we check if the x coordinate of the center point lies below that of the carbon
                elif P[0] > carbon_positions2D[indices[i][0]][0]:

                    if P[1] > carbon_positions2D[indices[i][0]][1]:
                        hydro.append([carbon_positions2D[indices[i][0]][0] - 0.8 * distance,
                                      carbon_positions2D[indices[i][0]][1] - 0.5 * distance, carbon_positions[i][-1]])

                    else:
                        hydro.append([carbon_positions2D[indices[i][0]][0] - 0.8 * distance,
                                      carbon_positions2D[indices[i][0]][1] + 0.5 * distance, carbon_positions[i][-1]])

                # first we check if the x coordinate of the center point lies below that of the carbon
                elif P[0] < carbon_positions2D[indices[i][0]][0]:

                    if P[1] > carbon_positions2D[indices[i][0]][1]:
                        hydro.append([carbon_positions2D[indices[i][0]][0] + 0.8 * distance,
                                      carbon_positions2D[indices[i][0]][1] - 0.5 * distance, carbon_positions[i][-1]])

                    else:
                        hydro.append([carbon_positions2D[indices[i]][0][0] + 0.8 * distance,
                                      carbon_positions2D[indices[i][0]][1] + 0.5 * distance, carbon_positions[i][-1]])

        return np.asarray(hydro)

    def create_save_nanographenes(self):

        # get current working directory
        my_path = os.path.dirname(__file__)
        save_dir = os.path.join(my_path, self.fold_date)
        pic_dir = os.path.join(save_dir, self.fold_date + "_draws")

        # make directory in which the xyz files should be stored
        os.makedirs(save_dir)
        os.makedirs(pic_dir)

        # create open shell directory
        open_shell_dir = os.path.join(my_path, self.fold_date, "open_shell")
        os.makedirs(open_shell_dir)

        # create closed shell directory
        closed_shell_dir = os.path.join(my_path, self.fold_date, "closed_shell")
        os.makedirs(closed_shell_dir)



        # counter for saving the molecules
        count = 0

        # function that builds and saves the nanographenes in a folder with the current time in xyz formate
        for conn in range(8, self.max_number_conon):

            # get all number of structures for this number of benzol rings

            if conn <= 3:
                structs = self.perform_random_walk_all(conn)
            else:
                structs = self.perform_random_walk_straight(conn)

            print("All structures are created fro ", conn, " connections")


            # Next layer of filtering is to create a metric, that if a




            for struct in structs:
                # derive the carbon and hydrogen positions
                carbon_positions = self.get_unique_atom_position(struct)

                # derive the positions of the hydrogen atoms based on the coordinates of the carbons
                hydrogen_positions = self.get_unique_hydrogen_positions(carbon_positions)

                # directory checking for even or odd number of electrons
                if len(hydrogen_positions) % 2 != 0:
                    # open shell calculations are defined by an uneven number of electrons
                    save_dir = os.path.join(my_path, self.fold_date, "open_shell")
                else:
                    # closed shell calculations are defined by an even number of electrons
                    save_dir = os.path.join(my_path, self.fold_date, "closed_shell")

                # create out direcotory containing the geometry in and xyz file
                final_save_dir = os.path.join(save_dir, 'C' + str(len(carbon_positions)) + '_H' +
                                              str(len(hydrogen_positions)))

                # get name of picture that needs to be saved
                pic_name = "geometry_" + 'C' + str(len(carbon_positions)) + '_H' + str(len(hydrogen_positions))

                # check if the direcotry has alrady been created for another system
                identify = 0
                flag = False


                # check if path really exists
                if os.path.isdir(final_save_dir):

                    # the directory already exists so we need to increase the counter
                    flag = True

                    while flag == True:
                        # create count
                        identify = identify + 1
                        final_save_dir = os.path.join(save_dir, 'C' + str(len(carbon_positions)) + '_H' +
                                                      str(len(hydrogen_positions)) + "_" + str(identify))

                        pic_name = "geometry_" + 'C' + str(len(carbon_positions)) + '_H' + str(len(hydrogen_positions)) \
                                   + "_" + str(identify)


                        if os.path.isdir(final_save_dir):
                            flag = True
                        else:
                            flag = False

                # create specific molecule direcotry
                os.mkdir(final_save_dir)

                # create out path
                out_path = os.path.join(final_save_dir, 'geometry.xyz')

                with open(out_path, 'w') as geo_file:

                    # first line says number of atoms
                    geo_file.write(str(len(carbon_positions) + len(hydrogen_positions)) + '\n')
                    # comments line is empty
                    geo_file.write('C' + str(len(carbon_positions)) + 'H' + str(len(hydrogen_positions)) + '\n')

                    # loop over entries to write coordinates and element
                    for coords in carbon_positions:
                        line = "C" + '          ' + str(coords[0]) + '          ' + str(coords[1]) + '          ' + str(
                            coords[2]) + '\n'
                        geo_file.write(line)

                    for coords in hydrogen_positions:
                        line = "H" + '          ' + str(coords[0]) + '          ' + str(coords[1]) + '          ' + str(
                            coords[2]) + '\n'

                        geo_file.write(line)

                self.plot_save_nanographenes(carbon_positions, pic_name, pic_dir)

                # close file
                geo_file.close()

                # turn xyz file into in file
                self.transform_xyz_in(final_save_dir)

                count = count + 1

    @staticmethod
    def plot_save_nanographenes(carbon_positions, pic_name, pic_dir):
        # Function that plots the benzol rings (carbon atoms) and saves the plot

        carbon_positions2D = np.delete(carbon_positions, -1, axis=1)

        # use nearest neighbour algorithm to get two nearest neighbours (3 because itself is the nearest)
        nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(carbon_positions2D)

        # get distances and indices
        distances, indices = nbrs.kneighbors(carbon_positions2D)

        # create graph object
        G = nx.Graph()

        for i in range(len(carbon_positions2D)):
            G.add_node(i, pos=carbon_positions2D[i])

        edges = []

        # rearange the endge list
        for i in range(len(indices)):
            edges.append([indices[i][0], indices[i][1]])
            edges.append([indices[i][0], indices[i][2]])

            if distances[i][3] < 2.0:
                edges.append([indices[i][0], indices[i][3]])

        # plot everything
        G.add_edges_from(edges)

        pos = nx.get_node_attributes(G, 'pos')

        nx.draw(G, pos, nodelist=(), linewidths=2.0)

        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')

        plt.savefig(os.path.join(pic_dir, (pic_name + ".eps")), format="eps")
        plt.savefig(os.path.join(pic_dir, (pic_name + ".png")), format="png")

        plt.close()
