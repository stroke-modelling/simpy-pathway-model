"""
Class for organising paths, directory and file names.
"""
import os  # For defining paths.


class Setup(object):
    """
    Global directory and file names for the pathway.

    All attributes are defined in __init__.
    The output directory attributes can be changed later using
    the method create_output_dir.
    """

    def __init__(self, *initial_data, **kwargs):

        # Directories:
        # (don't include slashes please)
        self.dir_data = 'data'
        self.dir_output_this_setup = 'output'
        self.dir_output_all_runs = 'output'
        self.dir_output = 'run'
        self.dir_data_geojson = 'data_geojson'
        # Keep a list of output directories, e.g. one directory for
        # each scenario:
        self.list_dir_output = []

        # Input file names:
        self.file_input_unit_services = 'stroke_unit_services.csv'
        self.file_input_travel_times = 'lsoa_travel_time_matrix_calibrated.csv'
        self.file_input_travel_times_inter_unit = (
            'inter_hospital_time_calibrated.csv')
        self.file_input_lsoa_regions = 'LSOA_regions.csv'
        self.file_input_hospital_info = 'stroke_hospitals_2022_regions.csv'
        self.file_input_admissions = 'admissions_2017-2019.csv'
        # Geojson files:
        self.file_geojson_lsoa = 'LSOA_(Dec_2011)_Boundaries_Super_Generalised_Clipped_(BSC)_EW_V3.geojson'
        self.file_geojson_ccg = 'Clinical_Commissioning_Groups_April_2019_Boundaries_EN_BGC_2022_-7963862461000886750.geojson'
        self.file_geojson_icb = 'ICB_JUL_2022_EN_BGC_V3_7901616774526941461.geojson'
        self.file_geojson_lad = 'LAD_Dec_2017_GCB_GB_2022_5230662237199919616.geojson'
        self.file_geojson_stp = 'STP_Apr_2019_GCB_in_England_2022_3138810296697318496.geojson'
        self.file_geojson_lhb = 'Local_Health_Boards_April_2020_WA_BGC_2022_94310626700012506.geojson'
        self.file_geojson_scn = 'SCN_Dec_2016_GCB_in_England_2022_8470122845735728627.geojson'
        self.file_geojson_rgn = 'Regions_December_2022_EN_BGC_4589208765943883498.geojson'

        # Output file names:
        # Units():
        self.file_national_unit_services = 'national_unit_services.csv'
        self.file_national_lsoa_travel = 'national_lsoa_travel_units.csv'
        self.file_national_transfer_units = 'national_transfer_units.csv'
        # Scenario():
        self.file_selected_stroke_units = 'selected_stroke_units.csv'
        self.file_selected_lsoas = 'selected_lsoas.csv'
        # Model():
        self.file_results_all = 'results_all.csv'
        self.file_results_summary_all = 'results_summary_all.csv'
        self.file_results_summary_by_admitting_unit = (
            'results_summary_by_admitting_unit.csv')
        # Map():
        self.file_selected_units_map = 'map_selected_units.jpg'
        self.file_drip_ship_map = 'map_catchment_dripship.jpg'
        self.file_mothership_map = 'map_catchment_mothership.jpg'
        self.file_msu_map = 'map_catchment_msu.jpg'

        # Overwrite default values
        # (can take named arguments or a dictionary)
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

        # Check if this top output directory exists,
        # rename if necessary, then create the dictionary:
        self.dir_output_all_runs = self.create_output_dir(
            self.dir_output_all_runs, path_to_dir=self.dir_output_this_setup)

    def create_output_dir(self, dir_output, delim='!', path_to_dir=None):
        """
        Create a directory for storing the output of this run.

        If the chosen directory name already exists, add a suffix
        "!1" or similar to make a new directory name and create that
        instead.

        Choose a delimiter that wouldn't normally go into a dir
        name just to make the naming easier.

        Inputs
        ------
        dir_output          - str. Requested name of the output directory
                              for this run of the model only.
        delim               - str. A character to split off the requested
                              directory name from the suffix that this
                              function adds to it.
        """
        def _iterate_dir_suffix(dir_output, delim):
            """
            Update string for dir!{x} to dir!{x+1}.
            """
            if len(dir_output) > 0:
                if (dir_output[-1] == '/') | (dir_output[-1] == '\\'):
                    # Remove final '/' or '\'
                    dir_output = dir_output[:-1]
                else:
                    # Don't change the name.
                    pass
            else:
                # The name is just an empty string.
                pass
            # Split the dir name by every delim:
            dir_parts = dir_output.split(delim)
            # # Make a single string of everything up to the final delim:
            # For now, assume that the delimiter doesn't appear
            # elsewhere in the file name.
            dir_start = dir_parts[0]
            # If delimiter would appear elsewhere, would need something
            # like this only need a way to tell the difference between
            # a name ending _1 and ending _useful_stuff.
            # dir_start = (
            #     delim.join(dir_parts.split(delim)[:-1])
            #     if len(dir_parts) > 1 else dir_output)
            if len(dir_parts) == 1:
                # No delim in it yet. Set up the suffix.
                suffix = 1
            else:
                # Increase the number after the delim.
                suffix = dir_parts[-1]
                try:
                    suffix = int(suffix)
                    suffix += 1
                except ValueError:
                    # The final part of the directory name is not
                    # a number.
                    suffix = 1
            # Update the directory name:
            dir_output = f'{dir_start}{delim}{suffix}'
            return dir_output

        if path_to_dir is None:
            path_to_dir = self.dir_output_all_runs
            subdir = True
        else:
            subdir = False

        # Check if output folder already exists:
        dir_output_this_run = os.path.join(
            path_to_dir, dir_output)

        # While the requested output folder already exists,
        # add a suffix or increase its number until there's a new name.
        while os.path.isdir(dir_output_this_run):
            dir_output = _iterate_dir_suffix(dir_output, delim)
            dir_output_this_run = os.path.join(
                path_to_dir, dir_output)

        # Create this directory:
        os.mkdir(dir_output_this_run)

        if subdir:
            # Add the output directory to the list:
            self.list_dir_output.append(dir_output_this_run)
            # Save to self
            # (and so overwrite any name that was there before):
            self.dir_output = dir_output_this_run

        # Return the name so that we can point the code
        # at this directory:
        return dir_output_this_run
