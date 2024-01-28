"""
Stick this in a new home eventually. Not a class yet.

crs reference:
+ EPSG:4326  - longitude / latitude.
+ CRS:84     - same as EPSG:4326.
+ EPSG:27700 - British National Grid (BNG).
"""
import numpy as np
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import os

from shapely import LineString  # For creating line geometry.


# ##########################
# ##### DATA WRANGLING #####
# ##########################
def import_geojson(setup: 'Setup', region_type: 'str'):
    """
    Import a geojson file as GeoDataFrame.

    The crs (coordinate reference system) is set to British National
    Grid.

    Inputs
    ------
    setup       - Setup() object. Contains attributes for paths to the
                  data directory and the geojson file names.
    region_type - str. Lookup name for selecting a geojson file.
                  This should be one of the column names from the
                  various regions files.

    Returns
    -------
    gdf_boundaries - GeoDataFrame. One row per region shape in the
                     file. Expect columns for region name and geometry.
    """
    # Select geojson file based on input region type:
    geojson_file_dict = {
        'LSOA11NM': setup.file_geojson_lsoa,
        'CCG19NM': setup.file_geojson_ccg,
        'ICB22NM': setup.file_geojson_icb,
        'LAD17NM': setup.file_geojson_lad,
        'STP19NM': setup.file_geojson_stp,
        'LHB20NM': setup.file_geojson_lhb,
        'SCN17NM': setup.file_geojson_scn,
        'RGN11NM': setup.file_geojson_rgn,
    }
    # n.b. current setup as of January 2024 is that the dict
    # keys match the column names in the LSOA_regions.csv
    # and similar reference files. The actual geojson files
    # definitely contain the same type of region, but could
    # be from a different year than the one listed here.

    # Import region file:
    dir_input = setup.dir_data_geojson
    file_input = geojson_file_dict[region_type]
    path_to_file = os.path.join(dir_input, file_input)
    gdf_boundaries = geopandas.read_file(path_to_file)
    # If crs is given in the file, geopandas automatically
    # pulls it through. Convert to National Grid coordinates:
    if gdf_boundaries.crs != 'EPSG:27700':
        gdf_boundaries = gdf_boundaries.to_crs('EPSG:27700')
    return gdf_boundaries


def make_gdf_selected_stroke_unit_coords(setup: 'Setup'):
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
    # Import selected stroke unit data:
    dir_input = setup.dir_output
    file_input = setup.file_selected_stroke_units
    path_to_file = os.path.join(dir_input, file_input)
    df_units = pd.read_csv(path_to_file)

    # Create coordinates:
    # Current setup means sometimes these columns have different names.
    # TO DO - fix that please! ---------------------------------------------------
    x = df_units['Easting']
    y = df_units['Northing']
    xy = df_units[['Easting', 'Northing']]
    crs = 'EPSG:27700'

    df_units['geometry'] = geopandas.points_from_xy(x, y)
    # Make a column of coordinates [x, y]:
    df_units['coords'] = xy.values.tolist()

    # Convert to GeoDataFrame:
    gdf_units = geopandas.GeoDataFrame(
        df_units, geometry=df_units['geometry'], crs=crs
    )
    # Convert to British National Grid coordinates if necessary:
    if crs != 'EPSG:27700':
        gdf_units = gdf_units.to_crs('EPSG:27700')
    return gdf_units


def make_series_regions_containing_selected_stroke_units(
        gdf_units, region_type):
    """# List of intended areas:"""
    intended_regions = gdf_units[region_type].drop_duplicates().values
    return intended_regions


def import_transfer_unit_data(setup: 'Setup'):
    """
    Import national transfer unit data.

    The file read is the output from Units() after the
    national stroke unit services have been updated.

    Inputs
    ------
    setup - Setup() object. Contains attributes for paths to the
            data directory and the geojson file names.

    Returns
    -------
    df - pd.DataFrame. One row per national stroke unit in the file.
         Columns are each stroke unit, its time to chosen MT unit,
         and the name of its chosen MT unit:
         ['from_postcode', 'time_nearest_MT', 'name_nearest_MT'].
    """
    dir_input = setup.dir_output
    file_input = setup.file_national_transfer_units
    path_to_file = os.path.join(dir_input, file_input)
    df = pd.read_csv(path_to_file)
    return df


def import_selected_lsoa(setup: 'Setup'):
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
    dir_input = setup.dir_output
    file_input = setup.file_selected_lsoas
    path_to_file = os.path.join(dir_input, file_input)
    df = pd.read_csv(path_to_file)
    return df


def import_lsoa_travel_data(setup: 'Setup'):
    """
    Import each LSOA's chosen stroke units and travel times.

    Inputs
    ------
    setup - Setup() object. Contains attributes for paths to the
            data directory and the geojson file names.

    Returns
    -------
    df - pd.DataFrame. Each LSOA's name, chosen IVT/MT/MSU units,
         and travel times to those units. For X in IVT, MT, MSU:
         ['LSOA11NM', 'LSOA11CD', 'time_nearest_X',
          'postcode_nearest_X', 'ssnap_name_nearest_X']
    """
    dir_input = setup.dir_output
    file_input = setup.file_national_lsoa_travel
    path_to_file = os.path.join(dir_input, file_input)
    df = pd.read_csv(path_to_file)
    return df


def keep_only_selected_units(
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


def copy_columns_from_dataframe(
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


def make_gdf_lines_to_transfer_units(setup):
    """
    WRITE ME
    """
    gdf_units = make_gdf_selected_stroke_unit_coords(setup)

    # Find MT transfer units for plotting lines between units:
    df_transfer = import_transfer_unit_data(setup)
    df_transfer = keep_only_selected_units(
        df_transfer, gdf_units,
        left_col='from_postcode', right_col='Postcode')
    # Copy over coordinates of each stroke unit...
    df_transfer = copy_columns_from_dataframe(
        df_transfer, gdf_units,
        cols_to_copy=['coords'],
        cols_to_rename_dict={'coords': 'unit_coords'},
        left_col='from_postcode', right_col='Postcode', how='right'
        )
    # ... and its MT transfer unit:
    df_transfer = copy_columns_from_dataframe(
        df_transfer, gdf_units,
        cols_to_copy=['coords'],
        cols_to_rename_dict={'coords': 'transfer_coords'},
        left_col='name_nearest_MT', right_col='Postcode', how='left')
    gdf_transfer = create_lines_from_coords(
        df_transfer, ['unit_coords', 'transfer_coords'])
    return gdf_transfer


def create_lines_from_coords(df, cols_with_coords):
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

    # Convert line coords to LineString objects:
    df['line_geometry'] = [
        LineString(coords) for coords in df['line_coords']]

    # Convert to GeoDataFrame:
    gdf = geopandas.GeoDataFrame(
        df, geometry=df['line_geometry']#, crs="EPSG:4326"
    )
    # TO DO - implement CRS explicitly ---------------------------------------------
    return gdf


def make_gdf_lsoa_boundaries(setup):
    df_lsoa = import_selected_lsoa(setup)

    # Find LSOA boundaries:
    gdf_boundaries_lsoa = import_geojson(setup, 'LSOA11NM')
    gdf_boundaries_lsoa = keep_only_selected_units(
        gdf_boundaries_lsoa, df_lsoa,
        left_col='LSOA11CD', right_col='LSOA11CD', how='right'
        )

    # Match LSOA with its chosen stroke unit.
    df_lsoa_travel = import_lsoa_travel_data(setup)
    df_lsoa_travel = keep_only_selected_units(
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


def make_gdf_boundaries_regions_containing_possible_lsoa(
        setup,
        col,
        series_regions_containing_units
        ):
    # Find regional boundaries for reference on the map.
    # Assume that every stroke unit has at least one LSOA in the same
    # region as it. (This should always happen!)
    # So select the regional boundaries to make sure every LSOA
    # is drawn:
    regions_to_plot = find_region_catchment_across_all_model_types(setup, col)

    gdf_boundaries = import_geojson(setup, col)
    gdf_boundaries = keep_only_selected_units(
        gdf_boundaries, regions_to_plot,
        left_col=col, right_col=col, how='right'
        )
    # Drop any missing values
    # TO DO - this happens due to mismatch of Welsh and English regions - need better linkage ------------
    gdf_boundaries = gdf_boundaries.dropna(subset=[col])

    # Split off regions that are explicitly requested
    # (they contain a selected stroke team)
    # from those that get swept in by the LSOA catchment area
    # (and do not contain a selected stroke team).

    # Set all regions to be unintented...
    gdf_boundaries['additional_region'] = True
    # ... and then find which ones were intended.
    region_mask = [
        gdf_boundaries[col].str.contains(region)
        for region in series_regions_containing_units
        ]
    region_mask = np.any(region_mask, axis=0)
    # Update the ones that were intended.
    gdf_boundaries.loc[region_mask, 'additional_region'] = False
    return gdf_boundaries


def find_region_catchment_across_all_model_types(setup, col='ICB22NM'):
    """
    Limit LSOAs to those whose nearest stroke units are in the list.

    Example: square island. We model only the MT unit, and the
    national data includes the IVT-only unit too.
    In the mothership model, the whole island is in the catchment
    area of the MT unit. However in the drip-and-ship model, only
    half of the island is in the catchment area of the MT unit.

            Drip-and-ship           Mothership
        +-------------+        +-------------+
        |~~o~~~~~~~~_/|        |  o          |    o - IVT-only unit
        |~~~~~~~~__/  |        |             |    * - MT unit
        |~~~~~__/     |        |             |
        |~~__/        |        |             |    ~ - not in MT unit
        |_/        *  |        |          *  |        catchment.
        +-------------+        +-------------+
    """
    # First find the LSOAs that will travel to any selected
    # stroke unit *for any destination model type*.
    # This means easier comparison of maps later between
    # different model types. The same regions will be plotted
    # in both even if some regions contain LSOAs that only
    # are included for some model types.

    # Take list of all LSOA names and travel times:
    dir_input = setup.dir_output
    file_input = setup.file_national_lsoa_travel
    path_to_file = os.path.join(dir_input, file_input)
    df_travel = pd.read_csv(path_to_file)
    # This has one row for each LSOA nationally and columns
    # for LSOA name and ONS code (LSOA11NM and LSOA11CD),
    # and time, postcode, and SSNAP name of the nearest unit
    # for each unit type (IVT, MT, MSU).
    # Columns:
    # + LSOA11NM
    # + LSOA11CD
    # + time_nearest_IVT
    # + postcode_nearest_IVT
    # + ssnap_name_nearest_IVT
    # + time_nearest_MT
    # + postcode_nearest_MT
    # + ssnap_name_nearest_MT
    # + time_nearest_MSU
    # + postcode_nearest_MSU
    # + ssnap_name_nearest_MSU

    # Which LSOAs are in the catchment areas for these IVT units?
    # For each stroke team, make a long list of True/False for
    # whether each LSOA has this as its nearest unit.
    postcode_cols = [
        'postcode_nearest_IVT',
        'postcode_nearest_MT',
        'postcode_nearest_MSU',
    ]
    # Assume that "hospitals" has "Postcode" as its index.
    lsoa_bool = [
        _find_lsoa_catchment_mask(setup, df_travel, col)
        for col in postcode_cols
        ]
    # Mask is True for any LSOA that is True in any of the
    # lists in lsoa_bool.
    mask = np.any(lsoa_bool, axis=0)
    # Limit the data to just these LSOAs:
    lsoas_to_include = df_travel['LSOA11NM'][mask]

    # Match these to the region names:
    dir_input = setup.dir_data
    file_input = setup.file_input_lsoa_regions
    path_to_file = os.path.join(dir_input, file_input)
    df_lsoa_regions = pd.read_csv(path_to_file)

    # Restrict big LSOA region file to only the included LSOA:
    df_lsoa_regions = keep_only_selected_units(
        df_lsoa_regions, lsoas_to_include,
        left_col='LSOA11NM', right_col='LSOA11NM', how='right'
        )

    # Take only the regions that contain these LSOA
    # and remove repeats:
    regions_to_plot = df_lsoa_regions[col].drop_duplicates()

    return regions_to_plot


def _find_lsoa_catchment_mask(setup, df_travel, col):
    """
    DUPLICATED from Scenario - find a better way to do this.
    """
    # Save output to output folder.
    dir_output = setup.dir_output
    file_name = setup.file_selected_stroke_units
    path_to_file = os.path.join(dir_output, file_name)
    hospitals = pd.read_csv(path_to_file, index_col='Postcode')

    # Which LSOAs are in the catchment areas for these units?
    # For each stroke team, make a long list of True/False for
    # whether each LSOA has this as its nearest unit.
    # Assume that "hospitals" has "Postcode" as its index.
    lsoa_bool = [df_travel[col].str.contains(s)
                 for s in hospitals.index.values]
    # Mask is True for any LSOA that is True in any of the
    # lists in lsoa_bool.
    mask = np.any(lsoa_bool, axis=0)
    return mask


def assign_colours_to_regions(gdf, region_type):

    colours = ['ForestGreen', 'LimeGreen', 'RebeccaPurple', 'Teal']

    # TO DO - this neighbours function isn't working right -----------------------------
    # currently says that Gloucestershire doesn't border Bristol or Bath regions -------
    gdf = find_neighbours_for_regions(gdf, region_type)
    gdf = gdf.sort_values('total_neighbours', ascending=False)

    neighbour_list = gdf[region_type].tolist()

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


def round_coordinates(df_geojson):
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


def find_neighbours_for_regions(df_geojson, col='ICG22NM'):
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

    def find_neighbours(df_new, col):
        df = df_new.copy()
        df['my_neighbors'] = [[]] * len(df)

        for index, row in df.iterrows():
            if isinstance(row['geometry'], shapely.geometry.polygon.Polygon): 
                neighbors = df[df.geometry.touches(row['geometry'])][col].tolist()
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

    df_geojson = round_coordinates(df_geojson)
    df_new = split_multipolygons(df_geojson)
    df_neighbours = find_neighbours(df_new, col)
    df = unsplit_multipolygons(df_geojson, df_neighbours)

    # Record the number of neighbours:
    df['total_neighbours'] = df['neighbour_list'].str.len()
    return df



# ####################
# ##### PLOTTING #####
# ####################
# n.b. The following functions mostly just use plt.plot()
# but are given different wrappers anyway for the sake of
# applying some kwargs automatically.
def draw_boundaries(ax, gdf, **kwargs):
    """
    Draw regions from a GeoDataFrame.

    Inputs
    ------
    ax     - pyplot axis. Where to draw the regions.
    gdf    - GeoDataFrame. Stores geometry to be plotted.
    kwargs - dict. Keyword arguments to pass to plt.plot().

    Returns
    -------
    ax - pyplot axis. Same as input but with regions drawn on.
    """
    # Draw the main map with colours (choropleth):
    gdf.plot(
        ax=ax,              # Set which axes to use for plot (only one here)
        antialiased=False,  # Avoids artifact boundry lines
        **kwargs
        )
    return ax


def scatter_ivt_units(ax, gdf):
    """
    Draw scatter markers for IVT stroke units.

    Inputs
    ------
    ax     - pyplot axis. Where to draw the scatter markers.
    gdf    - GeoDataFrame. Stores stroke unit coordinates and services.

    Returns
    -------
    ax - pyplot axis. Same as input but with scatter markers.
    """
    # Scatter marker for each hospital:
    gdf.plot(
        ax=ax,
        edgecolor='k',
        facecolor='w',
        markersize=50,
        marker='o',
        zorder=2
        )
    return ax


def scatter_mt_units(ax, gdf):
    """
    Draw scatter markers for MT stroke units.

    Inputs
    ------
    ax     - pyplot axis. Where to draw the scatter markers.
    gdf    - GeoDataFrame. Stores stroke unit coordinates and services.

    Returns
    -------
    ax - pyplot axis. Same as input but with scatter markers.
    """
    # Scatter marker star for MT units:
    mask = gdf['Use_MT'] == 1
    MT = gdf[mask]
    MT.plot(
        ax=ax,
        edgecolor='k',
        facecolor='y',
        markersize=300,
        marker='*',
        zorder=2
        )
    return ax


def scatter_msu_units(ax, gdf):
    """
    Draw scatter markers for MSU stroke units.

    Inputs
    ------
    ax     - pyplot axis. Where to draw the scatter markers.
    gdf    - GeoDataFrame. Stores stroke unit coordinates and services.

    Returns
    -------
    ax - pyplot axis. Same as input but with scatter markers.
    """
    # Scatter marker star for MT/MSU units:
    mask = gdf['Use_MSU'] == 1
    MSU = gdf[mask]
    MSU.plot(
        ax=ax,
        edgecolor='k',
        facecolor='orange',
        markersize=50,
        marker='s',
        zorder=2
        )
    return ax


def plot_lines_between_units(ax, gdf):
    """
    Draw lines from stroke units to their MT transfer units.

    Inputs
    ------
    ax     - pyplot axis. Where to draw the scatter markers.
    gdf    - GeoDataFrame. Stores LineString objects that connect
             each stroke unit to its MT transfer unit.

    Returns
    -------
    ax - pyplot axis. Same as input but with scatter markers.
    """
    # Draw a line connecting each unit to its MT unit.
    gdf.plot(
        ax=ax,
        edgecolor='k',
        linestyle='-',
        linewidth=3,
        zorder=1  # Place it beneath the scatter markers.
    )
    return ax


def annotate_unit_labels(ax, gdf):
    """
    Draw label for each stroke unit.

    Inputs
    ------
    ax     - pyplot axis. Where to draw the scatter markers.
    gdf    - GeoDataFrame. Stores coordinates and name of each
             stroke unit.

    Returns
    -------
    ax - pyplot axis. Same as input but with scatter markers.
    """
    try:
        mask = gdf['labels_mask']
        gdf_labels = gdf[mask]
    except KeyError:
        # No mask column was given.
        gdf_labels = gdf

    # Define "z" to shorten following "for" line:
    z = zip(
        gdf_labels.geometry.x,
        gdf_labels.geometry.y,
        gdf_labels.Hospital_name
        )
    for x, y, label in z:
        # Edit the label to put a space in the postcode when displayed:
        label = f'{label[:-3]} {label[-3:]}'
        # Place the label slightly offset from the
        # exact hospital coordinates (x, y).
        ax.annotate(
            label, xy=(x, y), xytext=(8, 8),
            textcoords="offset points",
            bbox=dict(facecolor='w', edgecolor='k'),
            fontsize=8
            )
    return ax


# ######################
# ##### MAIN PLOTS #####
# ######################
def plot_map_selected_units(setup, col='ICB22NM'):
    """
    Make map of the selected units and the regions containing them.

    Properties of this map:
    + Each stroke unit is shown with a scatter marker.
    + Non-MT units are shown as circles and MT units as stars.
    + Lines are drawn between each non-MT unit and its chosen MT unit.
    + Each stroke unit is labelled in an offset text box.
    + The regions that contain the selected units are drawn in
      the background with each region given a different colour from
      its neighbours. These regions have an outline.

    Required data files:
    + geojson file of choice.
      Must contain:
      + coordinates of each feature / region boundary shape.
    + selected stroke unit file
      Output from Scenario.
      Must contain:
      + Postcode
        - for unit name matching.
        - for labels on the map.
      + Use_MT
        - for scatter marker choice.
      + [region]
        - region names to match the geojson file, for limiting the
          plotted areas to just those containing the stroke units.
      + Easting, Northing
        - for placement of the scatter markers.
    + national transfer unit file
      Output from Units.
      + from_postcode
        - for unit name matching.
      + name_nearest_MT
        - for setting up lines drawn between each stroke unit and
          its nearest MT unit.

    Result is saved as the name given in setup.file_selected_units_map.
    """
    # ----- Setup -----

    # Stroke unit setup
    gdf_points_units = make_gdf_selected_stroke_unit_coords(setup)
    gdf_lines_transfer = make_gdf_lines_to_transfer_units(setup)

    # Import background region shapes:
    gdf_boundaries = import_geojson(setup, col)
    gdf_boundaries = keep_only_selected_units(
        gdf_boundaries, gdf_points_units[[col]].drop_duplicates(), left_col=col, right_col=col, how='right')
    gdf_boundaries = assign_colours_to_regions(gdf_boundaries, col)

    # ----- Plotting -----
    # Plot the map.
    # Make max dimensions XxY inch:
    fig, ax = plt.subplots(figsize=(10, 10))

    ax = draw_boundaries(
        ax, gdf_boundaries,
        color=gdf_boundaries['colour'],
        edgecolor='k')
    ax = scatter_ivt_units(ax, gdf_points_units)
    ax = scatter_mt_units(ax, gdf_points_units)
    ax = scatter_msu_units(ax, gdf_points_units)
    ax = plot_lines_between_units(ax, gdf_lines_transfer)
    ax = annotate_unit_labels(ax, gdf_points_units)

    ax.set_axis_off()  # Turn off axis line and numbers

    # Save output to output folder.
    dir_output = setup.dir_output
    file_name = setup.file_selected_units_map
    path_to_file = os.path.join(dir_output, file_name)
    plt.savefig(path_to_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_map_catchment(setup, col='ICB22NM'):
    """
    Map the selected units, containing regions, and catchment areas.

    Creates three maps.
    + "Drip & Ship" - catchment area of each IVT unit.
    + "Mothership" - catchment area of each MT unit, no IVT units.
    + "MSU" - catchment area of each MSU unit.

    Properties of all maps:
    + Each stroke unit is shown with a scatter marker.
    + Non-MT units are shown as circles and MT units as stars.
    + Lines are drawn between each non-MT unit and its chosen MT unit.
    + Each stroke unit is labelled in an offset text box.
    + The regions that contain the selected units are drawn in
      the background with each region given a different colour from
      its neighbours. These regions have an outline.

    Required data files:
    + geojson file of choice.
      Must contain:
      + coordinates of each feature / region boundary shape.
    + selected stroke unit file
      Output from Scenario.
      Must contain:
      + Postcode
        - for unit name matching.
        - for labels on the map.
      + Use_MT
        - for scatter marker choice.
      + [region]
        - region names to match the geojson file, for limiting the
          plotted areas to just those containing the stroke units.
      + Easting, Northing
        - for placement of the scatter markers.
    + national transfer unit file
      Output from Units.
      + from_postcode
        - for unit name matching.
      + name_nearest_MT
        - for setting up lines drawn between each stroke unit and
          its nearest MT unit.
    + geojson file of LSOA boundaries.
      Must contain:
      + coordinates of each LSOA boundary.
    + selected LSOA name file.
      Must contain:
      + column LSOA11CD, one row per selected LSOA.
    + national LSOA travel data.
      Must contain:
      + column LSOA11CD for name matching.
      + postcode_nearest_IVT
      + postcode_nearest_MT
      + postcode_nearest_MSU

    Result is saved as the name given in setup class:
    + file_drip_ship_map
    + file_mothership_map
    + file_msu_map
    """
    # Stroke unit setup
    gdf_points_units = make_gdf_selected_stroke_unit_coords(setup)
    gdf_lines_transfer = make_gdf_lines_to_transfer_units(setup)

    series_regions_containing_units = (
        make_series_regions_containing_selected_stroke_units(
            gdf_points_units, col))

    # LSOA setup
    gdf_boundaries_lsoa = make_gdf_lsoa_boundaries(setup)

    # Background regions setup
    gdf_boundaries_regions = (
        make_gdf_boundaries_regions_containing_possible_lsoa(
            setup,
            col,
            series_regions_containing_units
            ))

    # ----- Plotting setup -----
    data_dicts = {
        'Drip & Ship': {
            'file': setup.file_drip_ship_map,
            'boundary_kwargs': {
                'column': 'postcode_nearest_IVT',
                'cmap': 'Blues',
                'edgecolor': 'face'
                },
            'scatter_ivt': True,
            'scatter_mt': True,
            'scatter_msu': False,
            },
        'Mothership': {
            'file': setup.file_mothership_map,
            'boundary_kwargs': {
                'column': 'postcode_nearest_MT',
                'cmap': 'Blues',
                'edgecolor': 'face'
                },
            'scatter_ivt': False,
            'scatter_mt': True,
            'scatter_msu': False,
            },
        'MSU': {
            'file': setup.file_msu_map,
            'boundary_kwargs': {
                'column': 'postcode_nearest_MSU',
                'cmap': 'Blues',
                'edgecolor': 'face'
                },
            'scatter_ivt': False,
            'scatter_mt': False,
            'scatter_msu': True,
            },
    }

    # ----- Actual plotting -----
    for model_type, data_dict in zip(data_dicts.keys(), data_dicts.values()):
        # Plot the map.
        # Make max dimensions XxY inch:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(model_type)

        # LSOAs:
        ax = draw_boundaries(
            ax, gdf_boundaries_lsoa,
            **data_dict['boundary_kwargs']
            )

        # Regions containing LSOAs but not stroke units:
        gdf_boundaries_unintended = gdf_boundaries_regions.loc[
            gdf_boundaries_regions['additional_region'] == True]
        ax = draw_boundaries(
            ax, gdf_boundaries_unintended,
            facecolor='none', edgecolor='silver', linewidth=0.5, linestyle='--'
            )

        # Regions containing stroke units:
        gdf_boundaries_intended = gdf_boundaries_regions.loc[
            gdf_boundaries_regions['additional_region'] == False]
        ax = draw_boundaries(
            ax, gdf_boundaries_intended,
            facecolor='none', edgecolor='k', linewidth=0.5
            )

        # Stroke unit markers.
        # Keep track of which units to label in here:
        gdf_points_units['labels_mask'] = False
        if data_dict['scatter_ivt']:
            ax = scatter_ivt_units(ax, gdf_points_units)
            gdf_points_units.loc[
                gdf_points_units['Use_IVT'] == 1, 'labels_mask'] = True
        if data_dict['scatter_mt']:
            ax = scatter_mt_units(ax, gdf_points_units)
            gdf_points_units.loc[
                gdf_points_units['Use_MT'] == 1, 'labels_mask'] = True
        if data_dict['scatter_msu']:
            ax = scatter_msu_units(ax, gdf_points_units)
            gdf_points_units.loc[
                gdf_points_units['Use_MSU'] == 1, 'labels_mask'] = True

        # Transfer unit lines.
        # Check whether they need to be drawn:
        draw_lines_bool = (
            (data_dict['scatter_ivt'] & data_dict['scatter_mt']) |
            (data_dict['scatter_ivt'] & data_dict['scatter_msu'])
        )
        if draw_lines_bool:
            ax = plot_lines_between_units(ax, gdf_lines_transfer)

        # Stroke unit labels.
        ax = annotate_unit_labels(ax, gdf_points_units)

        ax.set_axis_off()  # Turn off axis line and numbers

        # Save output to output folder.
        dir_output = setup.dir_output
        file_name = data_dict['file']
        path_to_file = os.path.join(dir_output, file_name)
        plt.savefig(path_to_file, dpi=300, bbox_inches='tight')
        plt.close()
