"""
Class for organising paths etc.
"""
import os  # For defining paths.


class Setup(object):
    """
    Paths and that - TO DO WRITE ME
    """

    def __init__(self, *initial_data, **kwargs):

        # Directories:
        # (don't include slashes please)
        self.dir_data = 'data'
        self.dir_output_all_runs = 'output'
        self.dir_output = 'run'

        # Input file names:
        self.file_input_unit_services = 'stroke_unit_services.csv'
        self.file_input_travel_times = 'lsoa_travel_time_matrix_calibrated.csv'
        self.file_input_travel_times_inter_unit = (
            'inter_hospital_time_calibrated.csv')
        self.file_input_lsoa_regions = 'LSOA_regions.csv'
        self.file_input_hospital_info = 'stroke_hospitals_2022_regions.csv'
        self.file_input_admissions = 'admissions_2017-2019.csv'

        # Output file names:
        # Units():
        self.file_national_unit_services = 'national_unit_services.csv'
        self.file_national_lsoa_travel = 'national_lsoa_travel_units.csv'
        self.file_national_transfer_units = 'national_transfer_units.csv'
        # Scenario():
        self.file_selected_stroke_units = 'selected_stroke_units.csv'
        self.file_selected_lsoas = 'selected_lsoas.csv'
        # Model():

        # Overwrite default values
        # (can take named arguments or a dictionary)
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

        # Set up paths to files.
        # Make a new output folder for each run.
        # Check if the requested output folder name already exists.
        # If it does then choose another one.
        # Then create a directory for the output files to go into.
        self.dir_output = self._create_output_dir(
            self.dir_output_all_runs, self.dir_output)

    def _create_output_dir(self, dir_output_all_runs, dir_output, delim='!'):
        """
        Create a directory for storing the output of this run.

        If the chosen directory name already exists, add a suffix
        "!1" or similar to make a new directory name and create that
        instead.

        Choose a delimiter that wouldn't normally go into a dir
        name just to make the naming easier.

        Inputs
        ------
        dir_output_all_runs - str. Name of the main output directory
                              for all runs.
        dir_output          - str. Requested name of the output directory
                              for this run of the model only.
        delim               - str. A character to split off the requested
                              directory name from the suffix that this
                              function adds to it.
        """
        # Check if output folder already exists:
        dir_output_this_run = os.path.join(dir_output_all_runs, dir_output)

        # While the requested output folder already exists:
        while os.path.isdir(dir_output_this_run):
            if ((dir_output[-1] == '/') | (dir_output[-1] == '\\')):
                # Remove final '/' or '\'
                dir_output = dir_output[:-1]
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
            dir_output_this_run = os.path.join(
                dir_output_all_runs, dir_output)
        # Create this directory:
        os.mkdir(dir_output_this_run)
        # Return the name so that we can point the code
        # at this directory:
        return dir_output_this_run
