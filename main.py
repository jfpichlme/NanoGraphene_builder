from Random_Nano_creator import Creator


def main():

    # First we initialize an Object that defines how many Rings we need to add
    nano_graphenes_object = Creator(20, max_number_conn=4)

    # Next, the graphene grid is set up from which we "cut out" the nanographenes randomly
    nano_graphenes_object.set_up_grid()

    # The nanographenes are created
    nano_graphenes_object.create_graphene_structure()

    # The nanographenes ar saved, plotted and an hydrogen atom is added to each unpaired electron
    nano_graphenes_object.create_save_nanographenes()


if __name__ == "__main__":
    main()
