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

# TO MERGE MULTIINDEX, have to know number of levels to add to the gdf data
# --> add levels then merge. Levels depends on whether using combo or single data.
# Make sure that dropping upper levels would recreate the single data version.
# TO DO.

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
    def load_run_data(self, dir_data=None):
        """
        Load in data specific to these runs.
        """
        if dir_data is None:
            # Use combined data files.
            dir_data = self.setup.dir_output_all_runs
            dicts_data = {
                'df_units': {
                    'file': self.setup.file_combined_selected_stroke_units,
                    'header': [0, 1]
                    },
                'df_lsoa': {
                    'file': self.setup.file_combined_selected_lsoas,
                    'header': [0, 1]
                    },
                'df_regions': {
                    'file': self.setup.file_combined_selected_regions,
                    'header': [0, 1]
                    },
                'df_results_by_unit': {
                    'file': (
                        self.setup.file_combined_results_summary_by_admitting_unit
                        ),
                    'header': [0, 1, 2]
                    },
                'df_results_by_lsoa': {
                    'file': self.setup.file_combined_results_summary_by_lsoa,
                    'header': [0, 1, 2]
                    },
            }
        else:
            # Use files for the selected scenario only.
            dicts_data = {
                'df_units': {
                    'file': self.setup.file_selected_stroke_units,
                    'header': [0]
                    },
                'df_lsoa': {
                    'file': self.setup.file_selected_lsoas,
                    'header': [0]
                    },
                'df_regions': {
                    'file': self.setup.file_selected_regions,
                    'header': [0]
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

        for label, data_dict in dicts_data.items():
            # Make path to file:
            path_to_file = os.path.join(dir_data, data_dict['file'])
            try:
                # Specify header to import as a multiindex DataFrame.
                df = pd.read_csv(path_to_file, header=data_dict['header'])
                # Save to self:
                setattr(self, label, df)
            except FileNotFoundError:
                # TO DO - proper error message
                print(f'Cannot import {label} from {data_dict["file"]}')

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
        file_input = geojson_file_dict[region_type]
        path_to_file = os.path.join(dir_input, file_input)
        gdf_boundaries = geopandas.read_file(path_to_file)
        # If crs is given in the file, geopandas automatically
        # pulls it through. Convert to National Grid coordinates:
        if gdf_boundaries.crs != 'EPSG:27700':
            gdf_boundaries = gdf_boundaries.to_crs('EPSG:27700')
        return gdf_boundaries

    # ##########################
    # ##### DATA WRANGLING #####
    # ##########################
    def process_data(self):
        self.load_geometry_lsoa()
        self.load_geometry_regions()
        self.load_geometry_stroke_units()

    def load_geometry_lsoa(self):
        """
        Create GeoDataFrames of new geometry and existing DataFrames.
        """
        # Selected LSOA names, codes, coordinates.
        # ['LSOA11NM', 'LSOA11CD', 'LSOA11BNG_N', 'LSOA11BNG_E',
        #  'LSOA11LONG', 'LSOA11LAT']
        df_lsoa = self.df_lsoa
        # How many MultiIndex column levels are there?
        n_levels = df_lsoa.columns.nlevels
        match_col_lsoa = self.find_multiindex_col(
            df_lsoa.columns, 'LSOA11NM')

        # If they exist, merge in the results by LSOA.
        results_exist = False
        try:
            # If the file wasn't loaded, this gives AttributeError:
            df_lsoa_results = self.df_results_by_lsoa
            # How many MultiIndex column levels are there?
            n_levels_results = df_lsoa_results.columns.nlevels
            match_col_results = self.find_multiindex_col(
                df_lsoa_results.columns, 'lsoa')
            # Update condition:
            results_exist = True
        except AttributeError:
            # Cannot merge in the results.
            pass

        if results_exist:
            # If the column levels are different,
            # add another column level to the shorter DataFrame.
            if n_levels > n_levels_results:
                df_lsoa_results = self.make_more_column_rows(
                    df_lsoa_results,
                    n_levels,
                    top_row_str='any',
                    mid_row_str=''
                    )
                # Find the renamed column to match by:
                match_col_results = self.find_multiindex_col(
                    df_lsoa_results.columns, 'lsoa')
            elif n_levels < n_levels_results:
                n_levels = n_levels_results
                
                print(df_lsoa)
                df_lsoa = self.make_more_column_rows(
                    df_lsoa,
                    n_levels,
                    top_row_str='any',
                    mid_row_str=''
                    )
                print(df_lsoa)
                # Find the renamed column to match by:
                match_col_lsoa = self.find_multiindex_col(
                    df_lsoa.columns, 'LSOA11NM')
            # Merge the DataFrames.
            # Assume that the first column contains the same type
            # of info in both DataFrames.
            df_lsoa = pd.merge(
                df_lsoa, df_lsoa_results,
                left_on=match_col_lsoa,
                right_on=match_col_results,
                how='left'
            )
            print('\nMerged results')
            for c in df_lsoa.columns:
                print(c)

        # All LSOA shapes:
        gdf_boundaries_lsoa = self.import_geojson('LSOA11NM')
        n_levels_boundaries = gdf_boundaries_lsoa.columns.nlevels
        match_col_boundaries = self.find_multiindex_col(
            gdf_boundaries_lsoa.columns, 'LSOA11NM')
        # If the column levels are different,
        # add another column level to the shorter DataFrame.
        if n_levels > n_levels_boundaries:
            gdf_boundaries_lsoa = self.make_more_column_rows(
                gdf_boundaries_lsoa,
                n_levels,
                top_row_str='any',
                mid_row_str=''
                )
            # Find the renamed column to match by:
            match_col_boundaries = self.find_multiindex_col(
                gdf_boundaries_lsoa.columns, 'LSOA11NM')
        elif n_levels < n_levels_boundaries:
            n_levels = n_levels_boundaries
            df_lsoa = self.make_more_column_rows(
                df_lsoa,
                n_levels,
                top_row_str='any',
                mid_row_str=''
                )
            # Find the renamed column to match by:
            match_col_lsoa = self.find_multiindex_col(
                df_lsoa.columns, 'LSOA11NM')

        # Merge the geometry and LSOA data.
        # Assume that the first column contains the same type
        # of info in both DataFrames.
        # Restrict to only the LSOAs selected.
        gdf_boundaries_lsoa = pd.merge(
            gdf_boundaries_lsoa, df_lsoa,
            left_on=match_col_boundaries,
            right_on=match_col_lsoa,
            how='right'
        )
        print('\nMerged geometry')
        for c in gdf_boundaries_lsoa.columns:
            print(c)
        # TO DO - make column names consistent. -------------------------------------------

        self.gdf_boundaries_lsoa = gdf_boundaries_lsoa

    def load_geometry_regions(self):
        """
        Create GeoDataFrames of new geometry and existing DataFrames.
        """
        # Selected regions names and usage.
        # ['{region type}', 'contains_selected_unit',
        #  'contains_selected_lsoa']
        df_regions = self.df_regions
        n_levels = df_regions.columns.nlevels
        match_col_regions = self.find_multiindex_col(
            df_regions.columns, self.region_type)

        # All region polygons:
        gdf_boundaries_regions = self.import_geojson(self.region_type)
        n_levels_boundaries = gdf_boundaries_regions.columns.nlevels
        match_col_boundaries = self.find_multiindex_col(
            gdf_boundaries_regions.columns, self.region_type)

        # If the column levels are different,
        # add another column level to the shorter DataFrame.
        if n_levels > n_levels_boundaries:
            gdf_boundaries_regions = self.make_more_column_rows(
                gdf_boundaries_regions,
                n_levels,
                top_row_str='any',
                mid_row_str=''
                )
            # Find the renamed column to match by:
            match_col_boundaries = self.find_multiindex_col(
                gdf_boundaries_regions.columns, self.region_type)
        elif n_levels < n_levels_boundaries:
            n_levels = n_levels_boundaries
            df_regions = self.make_more_column_rows(
                df_regions,
                n_levels,
                top_row_str='any',
                mid_row_str=''
                )
            # Find the renamed column to match by:
            match_col_regions = self.find_multiindex_col(
                df_regions.columns, self.region_type)

        # Then merge in the geometry (polygon):
        # Restrict to selected regions:
        # Merge in region types
        # and limit the boundaries file to only selected regions.
        gdf_boundaries_regions = pd.merge(
            gdf_boundaries_regions,
            df_regions,
            left_on=match_col_boundaries,
            right_on=match_col_regions,
            how='right'
            )
        # TO DO - how to pick out that column?!
        # IndexSlice? df.loc[:, pd.IndexSlice[:, :, bottom_row]]
        # Know column name before adding extra rows, so find it after.

        self.gdf_boundaries_regions = self.assign_colours_to_regions(
            gdf_boundaries_regions, self.region_type)

    def load_geometry_stroke_units(self):
        """
        Create GeoDataFrames of new geometry and existing DataFrames.
        """
        # Selected stroke units names, coordinates, and services.
        df_units = self.df_units
        # Convert to geometry (point):
        self.gdf_points_units = self.make_gdf_selected_stroke_unit_coords(df_units)
        # Convert to geometry (line):
        self.gdf_lines_transfer = self.make_gdf_lines_to_transfer_units(df_units)

    # ############################
    # ##### HELPER FUNCTIONS #####
    # ############################
    def make_more_column_rows(
            self, df, n_levels, top_row_str='any', mid_row_str=''):
        """
        Add extra column headers to match the other DataFrame.
        """
        cols = df.columns
        n_levels_already = cols.nlevels

        if (type(cols[0]) == list) | (type(cols[0]) == tuple):
            # Convert all columns tuples into an ndarray:
            all_cols = np.array([[n for n in c] for c in cols]).T
        else:
            # No MultiIndex.
            all_cols = [cols.values]

        n_mid = max(0, n_levels - (1 + n_levels_already))
        new_headers = (
            [np.array([top_row_str] * len(cols))] +
            [np.array([mid_row_str] * len(cols))] * n_mid
            )
        for c in all_cols:
            new_headers += [np.array(c)]

        # Create new MultiIndex DataFrame from the original:
        df = pd.DataFrame(
            df.values,
            index=df.index,
            columns=new_headers
        )
        return df

    def find_multiindex_col(self, columns, target):
        if (type(columns[0]) == list) | (type(columns[0]) == tuple):
            # Convert all columns tuples into an ndarray:
            all_cols = np.array([[n for n in c] for c in columns])
        else:
            # No MultiIndex.
            all_cols = columns.values
        # Find where the grid matches the target string:
        inds = np.where(all_cols == target)
        # If more than one column, select the first.
        ind = inds[0][0]
        # Components of column containing the target:
        bits = all_cols[ind]
        bits_is_list = (type(columns[0]) == list) | (type(columns[0]) == tuple)
        # TO DO - make this generic arraylike ^
        # Convert to tuple for MultiIndex or str for single level.
        final_col = list((tuple(bits), )) if bits_is_list else bits
        return final_col

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

        match_col_east = self.find_multiindex_col(
            df_units.columns, 'Easting')
        match_col_north = self.find_multiindex_col(
            df_units.columns, 'Northing')
        
        # Create coordinates:
        # Current setup means sometimes these columns have different names.
        # TO DO - fix that please! ---------------------------------------------------
        x = df_units[match_col_east].values.reshape(len(df_units))
        y = df_units[match_col_north].values.reshape(len(df_units))
        crs = 'EPSG:27700'  # by definition for easting/northing.

        # Make a new column name like Easting and Northing:
        col_geo = tuple([b for b in match_col_east[0][:-1]] + ['geometry'])
        df_units[col_geo] = geopandas.points_from_xy(x, y)

        # Convert to GeoDataFrame:
        gdf_units = geopandas.GeoDataFrame(
            df_units, geometry=df_units[col_geo], crs=crs
        )
        # Convert to British National Grid coordinates if necessary:
        if crs != 'EPSG:27700':
            gdf_units = gdf_units.to_crs('EPSG:27700')
        return gdf_units

    def make_gdf_lines_to_transfer_units(self, df_transfer):
        """
        WRITE ME
        """
        match_col_east = self.find_multiindex_col(
            df_transfer.columns, 'Easting')
        match_col_north = self.find_multiindex_col(
            df_transfer.columns, 'Northing')
        
        match_col_east_mt = self.find_multiindex_col(
            df_transfer.columns, 'Easting_mt')
        match_col_north_mt = self.find_multiindex_col(
            df_transfer.columns, 'Northing_mt')

        # Make a new column name like Easting and Northing:
        col_unit = tuple([b for b in match_col_east[0][:-1]] + ['unit_coords'])
        col_tran = tuple([b for b in match_col_east[0][:-1]] + ['transfer_coords'])
        
        # Make a column of coordinates [x, y]:
        cols = [c if type(c) != list else c[0] for c in [match_col_east, match_col_north]]
        print(cols)
        xy = df_transfer[cols]
        df_transfer[col_unit] = xy.values.tolist()

        cols = [c if type(c) != list else c[0] for c in [match_col_east_mt, match_col_north_mt]]
        print(cols)
        xy = df_transfer[cols]
        df_transfer[col_tran] = xy.values.tolist()


        cols = [c if type(c) != list else c[0] for c in [col_unit, col_tran]]
        print(cols)
        gdf_transfer = self.create_lines_from_coords(
            df_transfer, cols)
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
        # Make a new column name like Easting and Northing:
        col_coord = tuple([b for b in cols_with_coords[0][:-1]] + ['line_coords'])
        col_geom = tuple([b for b in cols_with_coords[0][:-1]] + ['line_geometry'])
    
        # Combine multiple columns of coordinates into a single column
        # with a list of lists of coordinates:
        df[col_coord] = df[cols_with_coords].values.tolist()

        # Drop any duplicates:
        df = df.drop_duplicates(col_coord)

        # Convert line coords to LineString objects:
        df[col_geom] = [
            LineString(coords) for coords in df[col_coord]]

        # Convert to GeoDataFrame:
        gdf = geopandas.GeoDataFrame(
            df, geometry=df[col_geom]#, crs="EPSG:4326"
        )
        # TO DO - implement CRS explicitly ---------------------------------------------
        return gdf

    def assign_colours_to_regions(self, gdf, region_type):

        colours = ['ForestGreen', 'LimeGreen', 'RebeccaPurple', 'Teal']

        # Use any old colours as debug:
        np.random.seed(42)
        colour_arr = np.random.choice(colours, size=len(gdf))

        # Make a new column name like Easting and Northing:
        col_col = tuple([b for b in gdf.columns[0][:-1]] + ['colour'])

        # Add to the DataFrame:
        gdf[col_col] = colour_arr

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

        def find_neighbours(df_new, region_type):
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
