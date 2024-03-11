"""
Class for organising paths, directory and file names.
"""
import os  # For defining paths.
from importlib_resources import files  # For defining paths.
import yaml


class Setup(object):
    """
    Global directory and file names for the pathway.

    All attributes are defined in __init__.
    The output directory attributes can be changed later using
    the method create_output_dir.

    Directory setup:
    > dir_output_top
        > dir_output_all_scenarios
            > list_dir_scenario[0]
                + file_selected_regions
                + file_selected_units
                > pathway
                    + file_selected_regions
                    + file_selected_units
                    + file_national_transfer_units
                    + file_selected_transfer_units
                    + file_selected_lsoas
                    + file_selected_lsoa_catchment_nearest
                    + file_selected_lsoa_catchment_island
                    + file_selected_lsoa_admissions
                    + file_results_all
                    + file_results_summary_all
                    + file_results_summary_by_admitting_unit
                    + file_results_summary_by_lsoa
                > maps
                    + file_gdf_boundaries_regions
                    + file_gdf_points_units
                    + file_gdf_lines_transfer
                    + file_gdf_boundaries_lsoa
                    + {assorted map images.jpg}
            > list_dir_scenario[1]
                {similar contents to list_dir_scenario[0]}
            ...
            {up to list_dir_scenario[n]}
            ...
            > dir_output_combined
                + file_combined_selected_regions
                + file_combined_selected_units
                + file_combined_selected_transfer_units
                + file_combined_selected_lsoas
                + file_combined_selected_regions
                + file_combined_selected_lsoa_admissions
                + file_combined_results_summary_by_admitting_unit
                + file_combined_results_summary_by_lsoa
                > maps
                    + file_gdf_boundaries_regions
                    + file_gdf_points_units
                    + file_gdf_lines_transfer
                    + file_gdf_boundaries_lsoa
                    + {assorted map images.jpg}

    TO DO - easier to store all dirs in one dict, all files in another? ----------
    """

    def __init__(self, *initial_data, **kwargs):
        # Assume we always want a fresh directory for this Setup.
        self.create_new_dir_output_all_scenarios = True

        # Where is all of this going anyway?
        self.path_before_dir_output_top = '.'

        # Directories:
        # (don't include slashes please)
        # Reference data:
        self.dir_reference_data = 'data'
        self.dir_reference_data_geojson = 'data_geojson'

        # Path to scenario directory:
        self.dir_output_top = 'output'
        self.dir_output_all_scenarios = 'output_group'
        # The current scenario directory:
        self.dir_scenario = 'scenario'
        # Keep a list of scenario directories:
        self.list_dir_scenario = []
        # self.list_path_to_dir_scenario = []
        # Subdirs of the scenario directory:
        self.name_dir_output_pathway = 'pathway'
        self.name_dir_output_maps = 'maps'
        # Combined multiple scenarios:
        self.dir_output_combined = 'combined'

        # Reference data file names:
        self.file_input_regions = 'regions_ew.csv'
        self.file_input_unit_services = 'stroke_units_regions.csv'
        self.file_input_travel_times = 'lsoa_travel_time_matrix_calibrated.csv'
        self.file_input_travel_times_inter_unit = (
            'inter_hospital_time_calibrated.csv')
        self.file_input_lsoa_regions = 'regions_lsoa_ew.csv'
        self.file_input_admissions = 'admissions_2017-2019.csv'
        # Reference geometry files for Maps:
        self.file_input_hospital_coords = 'unit_postcodes_coords.csv'
        self.file_geojson_lsoa = ''.join([
            'LSOA_(Dec_2011)_Boundaries_Super_Generalised_Clipped_(BSC)',
            '_EW_V3.geojson'
        ])
        self.file_geojson_sibcl = (
            'SICBL_JUL_2022_EN_BUC_4104971945004813003.geojson')
        self.file_geojson_lhb = ''.join([
            'Local_Health_Boards_April_2020_WA_BGC_2022_',
            '94310626700012506.geojson'
        ])

        # File names that the pathway can save to:
        filenames = [
            # Input file names:
            # 'selected_regions',
            'selected_units',
            # Output file names:
            # Scenario():
            'selected_transfer_units',
            'selected_lsoa',
            # Model():
            'results_all',
            'results_summary_all',
            'results_summary_by_admitting_unit',
            'results_summary_by_lsoa',
            # Combine():
            # 'combined_selected_regions',
            'combined_selected_units',
            'combined_selected_transfer_units',
            'combined_selected_lsoas',
            'combined_results_summary_by_admitting_unit',
            'combined_results_summary_by_lsoa',
            # Map():
            'gdf_boundaries_regions',
            'gdf_points_units',
            'gdf_lines_transfer',
            'gdf_boundaries_lsoa',
        ]
        for f in filenames:
            # Set an attribute: self.file_{f} = '{f}.csv'
            setattr(self, f'file_{f}', f'{f}.csv')

        # Which LSOA catchment type are we using?
        # This gets updated later with either
        # self.selected_lsoa_catchment_nearest or
        # self.selected_lsoa_catchment_island.
        self.file_selected_lsoas = None

        # Overwrite default values
        # (can take named arguments or a dictionary)
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

        # Create the path to each directory.
        self.dir_output_top = os.path.join(
            self.path_before_dir_output_top,
            self.dir_output_top
            )
        self.dir_output_all_scenarios = os.path.join(
            self.dir_output_top,
            self.dir_output_all_scenarios
        )
        # Check the required directory structure
        # up to and excluding any scenario directories.
        # Does the top directory already exist?
        if os.path.exists(self.dir_output_top):
            pass
        else:
            os.mkdir(self.dir_output_top)

        # Does this dir already exist?
        if os.path.exists(self.dir_output_all_scenarios):
            # Flag to rename the dir so we don't overwrite it.
            self.rename_dir_output_all_scenarios = True
        else:
            # Don't need to change the name.
            self.rename_dir_output_all_scenarios = False

        if self.create_new_dir_output_all_scenarios:
            # Create the all scenarios dir:
            self.create_all_scenario_dir()

    def save_to_file(self):
        """Save the variable dict as a .yml file."""
        setup_vars = vars(self)

        dir_output = self.dir_output_all_scenarios
        file_output = 'setup.yml'
        file_setup_vars = os.path.join(dir_output, file_output)

        with open(file_setup_vars, 'w') as f:
            yaml.dump(setup_vars, f)

    def import_from_file(self, path_to_setup_file):
        """Import a .yml file and overwrite attributes here."""
        with open(path_to_setup_file, 'r') as f:
            setup_vars_imported = yaml.safe_load(f)

        for key, val in setup_vars_imported.items():
            setattr(self, key, val)

    # ###########################
    # ##### TRACK SCENARIOS #####
    # ###########################
    def update_scenario_list(self):
        """
        Set the main directory for this scenario.
        """
        # If it's not already in the list of scenario directories,
        # then add it:
        if self.dir_scenario not in self.list_dir_scenario:
            self.list_dir_scenario.append(self.dir_scenario)
            # self.list_path_to_dir_scenario.append(self.path_to_dir_scenario)

    def make_list_dir_scenario(self):
        """
        Overwrite self.list_dir_scenario with new list from dirs.
        """
        # Gather names of all dirs in dir_output_all_scenarios.
        list_dir_scenario = next(os.walk(self.dir_output_all_scenarios))[1]
        # Add the paths:
        list_dir_scenario = [
            os.path.join(self.dir_output_all_scenarios, d)
            for d in list_dir_scenario
        ]
        # Remove the combined dir if it's in there:
        if self.dir_output_combined in list_dir_scenario:
            list_dir_scenario.remove(self.dir_output_combined)
        # Save to self:
        self.list_dir_scenario = list_dir_scenario

    # ##############################
    # ##### DIRECTORY CREATION #####
    # ##############################
    def create_all_scenario_dir(self, delim='!'):
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
        if self.rename_dir_output_all_scenarios:
            # While the requested output folder already exists,
            # add a suffix or increase its number until there's a new name.
            dir_output_all_scenarios = self.dir_output_all_scenarios
            dir_output_top = os.path.split(dir_output_all_scenarios)[0]
            dir_output_end = os.path.split(dir_output_all_scenarios)[-1]
            while os.path.isdir(dir_output_all_scenarios):
                dir_output_end = self._iterate_dir_suffix(
                    dir_output_end, delim)
                dir_output_all_scenarios = os.path.join(
                    dir_output_top, dir_output_end)
            self.dir_output_all_scenarios = dir_output_all_scenarios
            # Update flag so this doesn't run again.
            self.rename_dir_output_all_scenarios = False
        else:
            pass

        if self.create_new_dir_output_all_scenarios:
            # Create top directory:
            os.mkdir(self.dir_output_all_scenarios)
            # Update flag so this doesn't run again.
            self.create_new_dir_output_all_scenarios = False
        else:
            pass

    def create_output_dir(
            self,
            dir_output,
            delim='!',
            path_to_dir=None,
            combined=False
            ):
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
        if path_to_dir is None:
            path_to_dir = self.dir_output_all_scenarios
            subdir = True
        else:
            subdir = False

        # Check if output folder already exists:
        dir_output_this_run = os.path.join(
            path_to_dir, dir_output)

        # While the requested output folder already exists,
        # add a suffix or increase its number until there's a new name.
        while os.path.isdir(dir_output_this_run):
            dir_output = self._iterate_dir_suffix(dir_output, delim)
            # dir_output_this_run = os.path.join(
            #     path_to_dir, dir_output)
            dir_output_this_run = dir_output

        # Create this directory:
        os.mkdir(dir_output_this_run)

        if combined:
            # Save to self
            # (and so overwrite any name that was there before):
            self.dir_output_combined = dir_output_this_run
        elif subdir:
            # Save to self
            # (and so overwrite any name that was there before):
            self.dir_scenario = dir_output_this_run
            # Add the output directory to the list:
            self.update_scenario_list()

        # Return the name so that we can point the code
        # at this directory:
        return dir_output_this_run

    def _iterate_dir_suffix(self, dir_output, delim):
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
