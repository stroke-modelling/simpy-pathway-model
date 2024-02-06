"""
Draw some maps using output files.

crs reference:
+ EPSG:4326  - longitude / latitude.
+ CRS:84     - same as EPSG:4326.
+ EPSG:27700 - British National Grid (BNG).
"""

"""
NOTES
# Always load all data every time.
# Only load it once.
# Select different DataFrame columns for different plots.
# Separate dataframes for geography?

# for dir in output_dir_list:
# load in these files,
# combine into a single dataframe, 
# if only one dir then no problem.
# Same variables for everything, use dir name to mask data and plot different bits.

# Don't save a column named "geometry" or the load won't work.
# ... so then what?
"""    
import numpy as np
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import os

from shapely import LineString  # For creating line geometry.

from classes.setup import Setup


class Map(object):
    """
    Combine files from multiple runs of the pathway.

    class Combine():

    TO DO - write me
    """
    def __init__(self, *initial_data, **kwargs):

        self.region_type = 'ICB22NM'

        # Overwrite default values
        # (can take named arguments or a dictionary)
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

        # If no setup was given, create one now:
        try:
            self.setup
        except AttributeError:
            self.setup = Setup()

    # ##########################
    # ##### FILE SELECTION #####
    # ##########################
    def load_combined_data(self):
        """
        Limit LSOAs to those whose nearest stroke units are in the list.

        Example catchment areas:
        The two right-hand units are in the selected regions
        and the two units on the left are national units,
        not modelled directly.

            ▓▓▓▓▓▓▓▓▓░░░░░░░░█████▒▒▒▒▒  <-- Drip-and-ship   +------------+
            ▏   *        o     o   *  ▕                      | * MT unit  |
            ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒  <-- Mothership      | o IVT unit |
                       -->▏ ▕<--                             +------------+
                       Difference

        The catchment area boundaries are halfway between adjacent units.

        In the mothership model, the right-hand MT unit's catchment area
        covers the whole right half of the rectangle.
        In the drip-and-ship model, some of that area is instead assigned
        to the out-of-area IVT unit in the centre of the rectangle.
        """
        # TO DO

        
        pass

    def load_run_data_combined(self):
        """
        Load in combo data specific to these runs.
        """
        dir_combo = self.setup.dir_output_all_runs

        combo_data = {
            'df_combo_units': {
                'file': self.setup.file_combined_selected_stroke_units,
                'header': [0, 1]
                },
            'df_combo_lsoa': {
                'file': self.setup.file_combined_selected_lsoas,
                'header': None
                },
            'df_combo_regions': {
                'file': self.setup.file_combined_selected_regions,
                'header': [0, 1]
                },
            'df_combo_results_by_unit': {
                'file': (
                    self.setup.file_combined_results_summary_by_admitting_unit
                    ),
                'header': [0, 1, 2]
                },
            'df_combo_results_by_lsoa': {
                'file': self.setup.file_combined_results_summary_by_lsoa,
                'header': [0, 1, 2]
                },
        }

        for label, data_dict in combo_data.items():
            # Make path to file:
            path_to_file = os.join(dir_combo, data_dict['file'])
            try:
                # Specify header to import as a multiindex DataFrame.
                df = pd.read_csv(path_to_file, header=data_dict['header'])
                # Save to self:
                setattr(self, label, df)
            except FileNotFoundError:
                # TO DO - proper error message
                print(f'Cannot import {label} from {data_dict["file"]}')

    def load_run_data_single(self, dir_data):
        """
        Load in combo data specific to these runs.
        """
        single_data = {
            'df_units': {
                'file': self.setup.file_selected_stroke_units,
                'header': None
                },
            'df_lsoa': {
                'file': self.setup.file_selected_lsoas,
                'header': None
                },
            'df_regions': {
                'file': self.setup.file_selected_regions,
                'header': None
                },
            'df_results_by_unit': {
                'file': (
                    self.setup.file_results_summary_by_admitting_unit
                    ),
                'header': [0, 1]
                },
            'df_results_by_lsoa': {
                'file': self.setup.file_results_summary_by_lsoa,
                'header': [0, 1]
                },
        }

        for label, data_dict in single_data.items():
            # Make path to file:
            path_to_file = os.join(dir_data, data_dict['file'])
            try:
                # Specify header to import as a multiindex DataFrame.
                df = pd.read_csv(path_to_file, header=data_dict['header'])
                # Save to self:
                setattr(self, label, df)
            except FileNotFoundError:
                # TO DO - proper error message
                print(f'Cannot import {label} from {data_dict["file"]}')

    def load_geometry(self):
        """
        Create GeoDataFrames of new geometry and existing DataFrames.
        """
        # Selected LSOA names, codes, coordinates.
        # ['LSOA11NM', 'LSOA11CD', 'LSOA11BNG_N', 'LSOA11BNG_E',
        #  'LSOA11LONG', 'LSOA11LAT']
        df_lsoa 
        # If it exists, merge in the results by LSOA:
        # Then merge in the geometry (polygon):

        # Selected regions names and usage.
        # ['{region type}', 'contains_selected_unit',
        #  'contains_selected_lsoa']
        df_regions 
        # Then merge in the geometry (polygon):
        

        # Selected stroke units names, coordinates, and services.
        df_units = self.df_units
        # Convert to geometry (point):
        gdf_units = self.make_gdf_selected_stroke_unit_coords(df_units)
        # Convert to geometry (line):
        gdf_transfer = self.make_gdf_lines_to_transfer_units(df_units)
        
    
    def process_data(self):
        # ----- Setup -----

        # Stroke unit setup
        self.gdf_points_units = self.make_gdf_selected_stroke_unit_coords()
        self.gdf_lines_transfer = self.make_gdf_lines_to_transfer_units()

        # Background regions setup
        self.gdf_boundaries_regions = self.import_geojson(self.region_type)
        df_selected_regions = self.import_selected_regions()
        # Merge in region types
        # and limit the boundaries file to only selected regions.
        self.gdf_boundaries_regions = self.copy_columns_from_dataframe(
            self.gdf_boundaries_regions, df_selected_regions,
            cols_to_copy=['contains_selected_unit', 'contains_selected_lsoa'],
            left_col=self.region_type, right_col=self.region_type, how='right'
            )

        self.gdf_boundaries_regions = self.assign_colours_to_regions(self.gdf_boundaries_regions, self.region_type)

        # LSOA setup
        self.gdf_boundaries_lsoa = self.make_gdf_lsoa_boundaries()

        # Outcomes setup
        self.df_outcomes_by_lsoa = self.import_lsoa_outcomes()
        # Merge into geographic data:
        self.gdf_boundaries_lsoa = self.copy_columns_from_dataframe(
            self.gdf_boundaries_lsoa, self.df_outcomes_by_lsoa,
            cols_to_copy=[
                'mRS shift_mean',
                'utility_shift_mean',
                'mRS 0-2_mean'
                ],
            cols_to_rename_dict={
                'mRS shift_mean': 'mRS shift',
                'utility_shift_mean': 'utility_shift',
                'mRS 0-2_mean': 'mRS 0-2'
            },
            left_col='LSOA11NM', right_col='lsoa', how='left'
            )

    # ##########################
    # ##### DATA WRANGLING #####
    # ##########################
    def import_geojson(self, region_type: 'str'):
        """
        Import a geojson file as GeoDataFrame.

        The crs (coordinate reference system) is set to British National
        Grid.

        Inputs
        ------
        setup       - Setup() object. Contains attributes for paths to the
                    data directory and the geojson file names.
        self.region_type - str. Lookup name for selecting a geojson file.
                    This should be one of the column names from the
                    various regions files.

        Returns
        -------
        gdf_boundaries - GeoDataFrame. One row per region shape in the
                        file. Expect columns for region name and geometry.
        """
        # Select geojson file based on input region type:
        geojson_file_dict = {
            'LSOA11NM': self.setup.file_geojson_lsoa,
            'CCG19NM': self.setup.file_geojson_ccg,
            'ICB22NM': self.setup.file_geojson_icb,
            'LAD17NM': self.setup.file_geojson_lad,
            'STP19NM': self.setup.file_geojson_stp,
            'LHB20NM': self.setup.file_geojson_lhb,
            'SCN17NM': self.setup.file_geojson_scn,
            'RGN11NM': self.setup.file_geojson_rgn,
        }
        # n.b. current setup as of January 2024 is that the dict
        # keys match the column names in the LSOA_regions.csv
        # and similar reference files. The actual geojson files
        # definitely contain the same type of region, but could
        # be from a different year than the one listed here.

        # Import region file:
        dir_input = self.setup.dir_data_geojson
        file_input = geojson_file_dict[self.region_type]
        path_to_file = os.path.join(dir_input, file_input)
        gdf_boundaries = geopandas.read_file(path_to_file)
        # If crs is given in the file, geopandas automatically
        # pulls it through. Convert to National Grid coordinates:
        if gdf_boundaries.crs != 'EPSG:27700':
            gdf_boundaries = gdf_boundaries.to_crs('EPSG:27700')
        return gdf_boundaries

    def make_gdf_selected_stroke_unit_coords(self, df_units):
        """
        Import GeoDataFrame of selected stroke unit data.

        The file read is the output from Scenario() after the
        national stroke units have been reduced to those selected.

        The file contains coordinates (Easting, Northing) and (long, lat)
        which are picked out here and converted into Point() objects.
        The crs (coordinate reference system) is set to British National
        Grid.

        Inputs
        ------
        setup - Setup() object. Contains attributes for paths to the
                data directory and the geojson file names.

        Returns
        -------
        gdf_units - GeoDataFrame. One row per selected stroke unit in
                    the file. Columns include unit name and geometry.
        """

        # Create coordinates:
        # Current setup means sometimes these columns have different names.
        # TO DO - fix that please! ---------------------------------------------------
        x = df_units['Easting']
        y = df_units['Northing']
        crs = 'EPSG:27700'

        df_units['geometry'] = geopandas.points_from_xy(x, y)

        # Convert to GeoDataFrame:
        gdf_units = geopandas.GeoDataFrame(
            df_units, geometry=df_units['geometry'], crs=crs
        )
        # Convert to British National Grid coordinates if necessary:
        if crs != 'EPSG:27700':
            gdf_units = gdf_units.to_crs('EPSG:27700')
        return gdf_units

    def import_lsoa_outcomes(self):
        """
        Import data on selected LSOAs.

        Inputs
        ------
        setup - Setup() object. Contains attributes for paths to the
                data directory and the geojson file names.

        Returns
        -------
        df - pd.DataFrame. Names, codes, coordinates of selected LSOAs.
            ['LSOA11NM', 'LSOA11CD', 'LSOA11BNG_N', 'LSOA11BNG_E',
            'LSOA11LONG', 'LSOA11LAT']
        """
        dir_input = self.setup.dir_output
        file_input = self.file_results_summary_by_lsoa
        path_to_file = os.path.join(dir_input, file_input)
        # Specify header to import as a multiindex DataFrame.
        df = pd.read_csv(path_to_file, header=[0, 1])

        first_column_name = df.columns[0][0]

        # Convert to normal index:
        df.columns = ["_".join(a) for a in df.columns.to_flat_index()]
        # Rename the first column which didn't have multiindex levels:
        df = df.rename(columns={df.columns[0]: first_column_name})
        return df

    def keep_only_selected_units(self,
            df: 'pd.DataFrame', df_units: 'pd.DataFrame',
            left_col: 'str', right_col: 'str', how: 'str' = 'right'
            ):
        """
        Wrapper for pd.merge() for limiting one df to rows in another.

        This function exists just so that the effect of it is clear
        from the function name without needing comments.

        Inputs
        ------
        df        - pd.DataFrame. Contains rows for many things we're
                    not interested in.
        df_units  - pd.DataFrame. Contains only rows of things we're
                    interested in.
        left_col  - str. Name of the column of "df" to match.
        right_col - str. Name of the column of "df_units" to match,
                    and the only column whose data will be kept in
                    the resulting DataFrame.
        how       - str. How to merge the DataFrames. Normally want
                    'right' to keep only the rows of df_units and
                    remove any row in df that is not in df_units.

        Returns
        -------
        df - pd.DataFrame. Reduced DataFrame.
        """
        if isinstance(df_units, pd.Series):
            df_to_merge = df_units
        elif isinstance(df_units, (pd.DataFrame, geopandas.GeoDataFrame)):
            df_to_merge = df_units[right_col]
        else:
            # This shouldn't happen!
            df_to_merge = df_units[right_col]
            # TO DO - some sort of error message, or more checks for what works here? -----------------
        df = pd.merge(
            df, df_to_merge,
            left_on=left_col, right_on=right_col,
            how=how
        )
        return df

    def copy_columns_from_dataframe(self,
            df_left: 'pd.DataFrame',
            df_right: 'pd.DataFrame',
            cols_to_copy: list,
            left_col: 'str', right_col: 'str', how: 'str' = 'right',
            cols_to_rename_dict: dict = {},
            ):
        """
        Copy data from one DataFrame into another. Wrapper for pd.merge().

        This function exists just so that the effect of it is clear
        from the function name without needing comments.

        Inputs
        ------
        df_left             - pd.DataFrame. Contains rows for many things
                            we're not interested in.
        df_right            - pd.DataFrame. Contains only rows of things
                            we're interested in.
        cols_to_copy        - list. Names of the columns from df_right
                            to copy over.
        left_col            - str. Name of the column of df_left to match.
        right_col           - str. Name of the column of df_right to match.
        how                 - str. How to merge the DataFrames.
        cols_to_rename_dict - dict. Rename the columns in df_left that
                            have been copied over from df_right.

        Returns
        -------
        df_left - pd.DataFrame. Same as the input df_left with the addition
                of copied columns from df_right.
        """
        # Copy data from df_right to df_left:
        df_left = pd.merge(
            df_left,
            df_right[[right_col] + cols_to_copy],
            left_on=left_col, right_on=right_col,
            how=how
        )
        # Rename columns in df_left:
        df_left = df_left.rename(columns=cols_to_rename_dict)
        return df_left

    def make_gdf_lines_to_transfer_units(self, df_transfer):
        """
        WRITE ME
        """
        # Make a column of coordinates [x, y]:
        xy = df_transfer[['Easting', 'Northing']]
        df_transfer['unit_coords'] = xy.values.tolist()

        xy = df_transfer[['Easting_mt', 'Northing_mt']]
        df_transfer['transfer_coords'] = xy.values.tolist()

        gdf_transfer = self.create_lines_from_coords(
            df_transfer, ['unit_coords', 'transfer_coords'])
        return gdf_transfer

    def create_lines_from_coords(self, df, cols_with_coords):
        """
        Convert DataFrame with coords to GeoDataFrame with LineString.

        Initially group coordinates from multiple columns into one:
        +--------+--------+       +------------------+
        |  col1  |  col2  |       |   line_coords    |
        +--------+--------+  -->  +------------------+
        | [a, b] | [c, d] |       | [[a, b], [c, d]] |
        +--------+--------+       +------------------+
        And convert the single column into shapely.LineString objects
        with associated crs. Then convert the input DataFrame into
        a GeoDataFrame with the new Line objects.

        Inputs
        ------
        df               - pd.DataFrame. Contains some coordinates.
        cols_with_coords - list. List of column names in df that contain
                        coordinates for making lines.

        Returns
        -------
        gdf - GeoDataFrame. The input df with the new Line
            geometry objects and converted to a GeoDataFrame.
        """
        # Combine multiple columns of coordinates into a single column
        # with a list of lists of coordinates:
        df['line_coords'] = df[cols_with_coords].values.tolist()

        # Drop any duplicates:
        df = df.drop_duplicates('line_coords')

        # Convert line coords to LineString objects:
        df['line_geometry'] = [
            LineString(coords) for coords in df['line_coords']]

        # Convert to GeoDataFrame:
        gdf = geopandas.GeoDataFrame(
            df, geometry=df['line_geometry']#, crs="EPSG:4326"
        )
        # TO DO - implement CRS explicitly ---------------------------------------------
        return gdf

    def make_gdf_lsoa_boundaries(self):
        df_lsoa = self.import_selected_lsoa()

        # Find LSOA boundaries:
        gdf_boundaries_lsoa = self.import_geojson('LSOA11NM')
        gdf_boundaries_lsoa = self.keep_only_selected_units(
            gdf_boundaries_lsoa, df_lsoa,
            left_col='LSOA11CD', right_col='LSOA11CD', how='right'
            )

        # Match LSOA with its chosen stroke unit.
        df_lsoa_travel = self.import_lsoa_travel_data()
        df_lsoa_travel = self.keep_only_selected_units(
            df_lsoa_travel, df_lsoa,
            left_col='LSOA11CD', right_col='LSOA11CD'
            )
        cols_to_keep = [
            'LSOA11CD', 'postcode_nearest_IVT',
            'postcode_nearest_MT', 'postcode_nearest_MSU'
            ]
        gdf_boundaries_lsoa = pd.merge(
            gdf_boundaries_lsoa,
            df_lsoa_travel[cols_to_keep],
            left_on='LSOA11CD', right_on='LSOA11CD',
        )
        return gdf_boundaries_lsoa

    def assign_colours_to_regions(self, gdf, region_type):

        colours = ['ForestGreen', 'LimeGreen', 'RebeccaPurple', 'Teal']

        # Use any old colours as debug:
        np.random.seed(42)
        colour_arr = np.random.choice(colours, size=len(gdf))

        # Add to the DataFrame:
        gdf['colour'] = colour_arr

        return gdf

    def assign_colours_to_regions_BROKEN(self, gdf, region_type):

        colours = ['ForestGreen', 'LimeGreen', 'RebeccaPurple', 'Teal']

        # TO DO - this neighbours function isn't working right -----------------------------
        # currently says that Gloucestershire doesn't border Bristol or Bath regions -------
        gdf = self.find_neighbours_for_regions(gdf, self.region_type)
        gdf = gdf.sort_values('total_neighbours', ascending=False)

        neighbour_list = gdf[self.region_type].tolist()

        neighbour_grid = np.full((len(gdf), len(gdf)), False)
        for row, neighbour_list_here in enumerate(gdf['neighbour_list']):
            for n in neighbour_list_here:
                col = neighbour_list.index(n)
                neighbour_grid[row, col] = True
                neighbour_grid[col, row] = True

        # Make a grid. One column per colour, one row per region.
        colour_grid = np.full((len(gdf), len(colours)), True)
        # To index row x: colour_grid[x, :]
        # To index col x: colour_grid[:, x]

        for row, region in enumerate(neighbour_list):
            # Which colours can this be?
            colours_here = colour_grid[row, :]

            # Pick the first available colour.
            ind_to_pick = np.where(colours_here == True)[0][0]

            # Update its neighbours' colour information.
            rows_neighbours = np.where(neighbour_grid[row, :] == True)[0]
            # Only keep these rows when we haven't checked them yet:
            rows_neighbours = [r for r in rows_neighbours if r > row]
            colour_grid[rows_neighbours, ind_to_pick] = False

            # Update its own colour information.
            colour_grid[row, :] = False
            colour_grid[row, ind_to_pick] = True

        # Use the bool colour grid to assign colours:
        colour_arr = np.full(len(neighbour_list), colours[0], dtype=object)
        for i, colour in enumerate(colours):
            colour_arr[np.where(colour_grid[:, i] == True)] = colour

        # Add to the DataFrame:
        gdf['colour'] = colour_arr

        # Use any old colours as debug:
        # np.random.seed(42)
        # colour_arr = np.random.choice(colours, size=len(gdf))

        return gdf

    def round_coordinates(self, df_geojson):
        from shapely.geometry import shape, mapping  # For conversion from shapely polygon to geojson and back
        # Remove floating point errors
        for i in range(len(df_geojson)):
            poly = df_geojson['geometry'][i]
            # Convert shapely object to geojson object
            gpoly = mapping(poly)
            if len(gpoly['coordinates']) == 1:
                # This is probably a normal polygon.
                a_coords = np.array(gpoly['coordinates'])
                new_coords = np.round(a_coords, 3)
            else:
                # This is probably a multipolygon but could be a polygon
                # that has multiple sets of coordinates for some reason
                # (maybe a hole, doughnut-shaped polygon?).
                new_coords = []
                for c, coords in enumerate(gpoly['coordinates']):
                    a_coords = np.array(gpoly['coordinates'][c])
                    a_coords = np.round(a_coords, 3)
                    new_coords.append(a_coords)
            gpoly['coordinates'] = new_coords

            # Convert back to shapely object
            poly = shape(gpoly)
            # Place back into the DataFrame
            df_geojson['geometry'][i] = poly
            return df_geojson

    def find_neighbours_for_regions(self, df_geojson, region_type='ICG22NM'):
        import shapely

        def split_multipolygons(df_geojson):
            # Expand the dataframe - need a separate row for each polygon, so split apart any multipolygons.
            df = df_geojson.copy()
            df_new = pd.DataFrame(columns=df.columns)

            r = 0
            for i in range(len(df)):
                row = df.iloc[i]
                if isinstance(row['geometry'], shapely.geometry.polygon.Polygon):
                    # All ok here, copy row contents exactly:
                    df_new.loc[r] = df.iloc[i]
                    r += 1
                else:
                    # MultiPolygon! Split it.
                    # Place each Polygon on its own row.
                    multipoly = row['geometry']
                    for poly in list(multipoly.geoms):
                        row_new = row.copy()
                        row_new['geometry'] = poly
                        df_new.loc[r] = row_new
                        r += 1

            # Convert this to a GeoDataFrame to match the input df_geojson:
            df_new = geopandas.GeoDataFrame(
                df_new, geometry=df_new['geometry']#, crs="EPSG:4326"
            )
            return df_new

        def find_neighbours(df_new, self.region_type):
            df = df_new.copy()
            df['my_neighbors'] = [[]] * len(df)

            for index, row in df.iterrows():
                if isinstance(row['geometry'], shapely.geometry.polygon.Polygon): 
                    neighbors = df[df.geometry.touches(row['geometry'])][region_type].tolist()
                elif not isinstance(row['geometry'], shapely.geometry.point.Point):
                    # This is a MultiPolygon. Check each Polygon separately.
                    multipoly = row['geometry']
                    neighbors = []
                    for polygon in list(multipoly.geoms):
                        neighbours_here = df[df.geometry.intersects(polygon)].index.tolist()
                        neighbors += neighbours_here
                else:
                    # It's a point! Ignore.
                    pass
                try:
                    # Don't let the place be its own neighbour.
                    neighbors = neighbors.remove(row.name)
                except ValueError:
                    # Its own name is not in the list of neighbours.
                    pass
                df.at[index, 'my_neighbors'] = neighbors
                # df.loc[index]['my_neighbors'] = neighbors #", ".join([f'{n}' for n in neighbors])

            df_neighbours = df.copy()
            return df_neighbours

        def unsplit_multipolygons(df_geojson, df_neighbours):
            df = df_geojson.copy()
            df['neighbour_list'] = [[]] * len(df)

            for i in range(len(df)):
                # What is this called in the original dataframe?
                objectid = df.iloc[i]['OBJECTID']
                # region_name = df.iloc[i]['CCG19CD']
                # Where did this end up in the polygon dataframe?
                df_here = df_neighbours[df_neighbours['OBJECTID'] == objectid]
                # Combine multiple lists of neighbours into one list:
                list_of_neighbours = np.concatenate(df_here['my_neighbors'].values)
                # Remove any repeats:
                list_of_neighbours = list(set(list_of_neighbours))
                df.at[i, 'neighbour_list'] = list_of_neighbours
            return df

        df_geojson = self.round_coordinates(df_geojson)
        df_new = split_multipolygons(df_geojson)
        df_neighbours = find_neighbours(df_new, self.region_type)
        df = unsplit_multipolygons(df_geojson, df_neighbours)

        # Record the number of neighbours:
        df['total_neighbours'] = df['neighbour_list'].str.len()
        return df
