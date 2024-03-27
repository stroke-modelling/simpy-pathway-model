"""
Draw some maps using GeoDataFrames from the Catchment data.

crs reference:
+ EPSG:4326  - longitude / latitude.
+ CRS:84     - same as EPSG:4326.
+ EPSG:27700 - British National Grid (BNG).
"""
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon  # For extent box.
import geopandas
import numpy as np

import classes.plot_functions as maps  # for plotting.
from classes.utils import find_multiindex_column_names


# ##########################
# ##### DATA WRANGLING #####
# ##########################
def main(
        gdf_boundaries_regions,
        gdf_points_units,
        gdf_boundaries_catchment,
        gdf_boundaries_lsoa,
        gdf_lines_transfer,
        crop_axis_leeway=2000,
        colour_list_units=[],
        colour_list_periphery_units=[],
        ):
    """
    Set up GeoDataFrames for plotting - crop and add colours.

    Inputs
    ------
    gdf_boundaries_regions      - GeoDataFrame. Regions info.
    gdf_points_units            - GeoDataFrame. Units info.
    gdf_boundaries_catchment    - GeoDataFrame. Catchment areas.
    gdf_boundaries_lsoa         - GeoDataFrame. LSOA info.
    gdf_lines_transfer          - GeoDataFrame. Transfer info.
    crop_axis_leeway            - float. Padding space around the edge of
                                  the plots from the outermost thing of
                                  interest. Units to match gdf units.
    colour_list_units           - list. List of #rrggbbaa colour strings
                                  or colour maps for catchment areas.
    colour_list_periphery_units - list. List of #rrggbbaa colour strings
                                  or colour maps for periphery unit
                                  catchment areas.

    Returns
    -------
    gdf_boundaries_regions   - GeoDataFrame. Input but cropped.
    gdf_points_units         - GeoDataFrame. Input but cropped and
                               assigned colour strings.
    gdf_boundaries_catchment - GeoDataFrame. Input but cropped and
                               assigned colour strings.
    gdf_boundaries_lsoa      - GeoDataFrame. Input but cropped.
    gdf_lines_transfer       - GeoDataFrame. Input but cropped and
                               assigned colour strings.
    box_shared               - polygon. The box we cropped gdfs to.
    map_extent_shared        - list. [minx, maxx, miny, maxy]
                               axis coordinates.
    )
    """
    tup = crop_data_to_shared_extent(
        gdf_boundaries_regions,
        gdf_points_units,
        gdf_boundaries_catchment,
        gdf_boundaries_lsoa,
        gdf_lines_transfer,
        leeway=crop_axis_leeway
        )
    (gdf_boundaries_regions,
     gdf_points_units,
     gdf_boundaries_catchment,
     gdf_boundaries_lsoa,
     gdf_lines_transfer,
     box_shared,
     map_extent_shared
     ) = tup
    # These GeoDataFrames will contain fewer rows than the files they
    # were created from because of the cropping - e.g. lots of stroke
    # units have been removed because they sit outside the map.

    # For each scenario, create colours for unit catchment areas
    # and their transfer units.
    scenario_list = list(set(
        gdf_boundaries_catchment.columns.get_level_values('scenario')))

    col_unit = find_multiindex_column_names(
        gdf_points_units, property=['unit'])
    col_postcode = find_multiindex_column_names(
        gdf_lines_transfer, property=['postcode'])

    # Define index columns so that pd.merge can be used
    # (it doesn't behave well with MultiIndex merging).
    gdf_lines_transfer = gdf_lines_transfer.set_index(col_postcode)
    gdf_points_units = gdf_points_units.set_index(col_unit)
    gdf_points_units.index.name = 'unit'

    for scenario in scenario_list:
        skip_this = (
            (scenario == '') |
            (scenario == 'any') |
            (scenario.startswith('Unnamed'))
        )
        if skip_this:
            pass
        else:
            col_output_colour_lines = ('colour_lines', scenario)
            col_use = ('use', scenario)

            # Convert the colour index columns in the gdf into
            # #rrggbbaa strings that pyplot understands.
            gdf_boundaries_catchment = make_colours_for_catchment(
                gdf_boundaries_catchment,
                colour_list_units,
                colour_list_periphery_units,
                col_colour_ind=('colour_ind', scenario),
                col_transfer_colour_ind=('transfer_colour_ind', scenario),
                col_selected=('selected', scenario),
                col_use=col_use,
                col_output_colour=('colour', scenario),
                col_output_colour_lines=('colour_lines', scenario),
                col_output_colour_periphery=('colour_periphery', scenario)
                )
            col_unit = find_multiindex_column_names(
                gdf_boundaries_catchment, property=['unit'])
            gdf_boundaries_catchment = (
                gdf_boundaries_catchment.set_index(col_unit))
            mask = (gdf_boundaries_catchment[col_use] == 1)

            # Move colours over to the transfer unit gdf.
            gdf_lines_transfer = pd.merge(
                gdf_lines_transfer,
                gdf_boundaries_catchment.loc[mask, [col_output_colour_lines]],
                left_index=True,
                right_index=True,
                how='left'
                )
            # Move colours over to the unit scatter markers gdf.
            # Add a blank 'subtype' level to catchment:
            gdf_here = gdf_boundaries_catchment.copy().loc[
                mask, [col_output_colour_lines]]
            gdf_here.columns = [
                gdf_here.columns.get_level_values('property'),
                gdf_here.columns.get_level_values('scenario'),
                [''] * len(gdf_here.columns)
            ]
            gdf_here.columns = gdf_here.columns.set_names(
                ['property', 'scenario', 'subtype'])
            gdf_here.index.name = 'unit'

            gdf_points_units = pd.merge(
                gdf_points_units,
                gdf_here,
                left_index=True,
                right_index=True,
                how='left'
                )

            gdf_boundaries_catchment = gdf_boundaries_catchment.reset_index()

    # Put the index columns back to how they started:
    gdf_lines_transfer = gdf_lines_transfer.reset_index()
    gdf_points_units = gdf_points_units.reset_index()

    return (
        gdf_boundaries_regions,
        gdf_points_units,
        gdf_boundaries_catchment,
        gdf_boundaries_lsoa,
        gdf_lines_transfer,
        box_shared,
        map_extent_shared
        )


def crop_data_to_shared_extent(
        gdf_boundaries_regions,
        gdf_points_units,
        gdf_boundaries_catchment,
        gdf_boundaries_lsoa,
        gdf_lines_transfer=None,
        leeway=10000
        ):
    """
    Only keep geometry in a shared bounding box.

    For example, the units gdf is reduced to only units within the box.
    The region boundaries are reduced to only regions that are at least
    partly in the box and then the region boundaries are trimmed to the
    box edges.

    Inputs
    ------
    gdf_boundaries_regions   - GeoDataFrame. Regions info.
    gdf_points_units         - GeoDataFrame. Units info.
    gdf_boundaries_catchment - GeoDataFrame. Catchment areas.
    gdf_boundaries_lsoa      - GeoDataFrame. LSOA info.
    gdf_lines_transfer       - GeoDataFrame. Transfer info.
    leeway                   - float. Padding space around the edge of
                               the plots from the outermost thing of
                               interest. Units to match gdf units.

    Returns
    -------
    gdf_boundaries_regions   - GeoDataFrame. Input and cropped.
    gdf_points_units         - GeoDataFrame. Input and cropped.
    gdf_boundaries_catchment - GeoDataFrame. Input and cropped.
    gdf_boundaries_lsoa      - GeoDataFrame. Input and cropped.
    gdf_lines_transfer       - GeoDataFrame. Input and cropped.
    box                      - polygon. The box we cropped gdfs to.
    map_extent               - list. [minx, maxx, miny, maxy]
                               axis coordinates.
    """
    # Find a shared axis extent for all GeoDataFrames.
    # Draw a box that just contains everything useful in all gdf,
    # then extend it a little bit for a buffer.
    box, map_extent = find_shared_map_extent(
        gdf_boundaries_regions,
        gdf_points_units,
        gdf_boundaries_catchment,
        gdf_lines_transfer,
        leeway
        )

    # Restrict all gdf to only geometry that falls within this box.
    gdf_boundaries_regions = _keep_only_geometry_in_box(
        gdf_boundaries_regions, box)
    gdf_points_units = _keep_only_geometry_in_box(
        gdf_points_units, box)
    gdf_boundaries_catchment = _keep_only_geometry_in_box(
        gdf_boundaries_catchment, box)
    gdf_boundaries_lsoa = _keep_only_geometry_in_box(
        gdf_boundaries_lsoa, box)

    if gdf_lines_transfer is None:
        pass
    else:
        gdf_lines_transfer = _keep_only_geometry_in_box(
            gdf_lines_transfer, box)

    # Crop any region boundaries that might extend outside the box.
    gdf_boundaries_regions = _restrict_geometry_edges_to_box(
        gdf_boundaries_regions, box)
    gdf_boundaries_catchment = _restrict_geometry_edges_to_box(
        gdf_boundaries_catchment, box)

    # Create labels *after* choosing the map
    # extent and restricting the regions to the edges of the box.
    # Otherwise labels could appear outside the plot.
    gdf_boundaries_regions = _assign_label_coords_to_regions(
        gdf_boundaries_regions, 'point_label')

    to_return = (
        gdf_boundaries_regions,
        gdf_points_units,
        gdf_boundaries_catchment,
        gdf_boundaries_lsoa,
        gdf_lines_transfer,
        box,
        map_extent
        )
    return to_return


def filter_gdf_by_columns(gdf, col_names):
    """
    Only take selected columns and rows where any value is 1.

    Use this to only keep certain parts of the GeoDataFrame,
    e.g. selected units.

    Inputs
    ------
    gdf - GeoDataFrame. To be filtered.
    col_names = list. Column names to keep.

    Returns
    -------
    gdf_reduced - GeoDataFrame. The requested subset of values.
    """
    # Which columns do we want?
    cols = gdf.columns.get_level_values('property').isin(col_names)
    # Subset of only these columns:
    gdf_selected = gdf.loc[:, cols]
    # Mask rows where any of these columns equal 1:
    mask = (gdf_selected == 1).any(axis='columns')
    # Only keep those rows:
    gdf_reduced = gdf.copy()[mask]
    return gdf_reduced


def find_shared_map_extent(
        gdf_boundaries_regions,
        gdf_points_units,
        gdf_boundaries_catchment,
        gdf_lines_transfer=None,
        leeway=None
        ):
    """
    Find axis boundaries that show the interesting parts of the gdfs.

    Take map extent from the combined LSOA, region, and stroke unit
    geometry.

    This assumes that all input gdf share a crs.

    Inputs
    ------
    gdf_boundaries_regions   - GeoDataFrame. Regions info.
    gdf_points_units         - GeoDataFrame. Units info.
    gdf_boundaries_catchment - GeoDataFrame. Catchment areas.
    gdf_boundaries_lsoa      - GeoDataFrame. LSOA info.
    gdf_lines_transfer       - GeoDataFrame. Transfer info.
    leeway                   - float. Padding space around the edge of
                               the plots from the outermost thing of
                               interest. Units to match gdf units.

    Returns
    -------
    box                      - polygon. The box we'll crop gdfs to.
    map_extent               - list. [minx, maxx, miny, maxy]
                               axis coordinates.
    """
    gdfs_to_merge = []

    def filter_gdf(gdf, cols_to_filter):
        # Pick out only rows where any cols are 1:
        gdf = filter_gdf_by_columns(gdf, cols_to_filter)
        # Keep only the 'geometry' column and name it 'geometry'
        # (no MultiIndex):
        gdf = gdf.reset_index()['geometry']
        gdf.columns = ['geometry']
        # Set the geometry column again:
        gdf = gdf.set_geometry('geometry')
        return gdf

    # Regions.
    # Pick out regions containing units:
    gdf_regions_reduced = filter_gdf(gdf_boundaries_regions, ['contains_unit'])
    gdfs_to_merge.append(gdf_regions_reduced)

    # Units.
    # Pick out units that have been selected or catch periphery LSOA.
    gdf_units_reduced = filter_gdf(
        gdf_points_units, ['selected', 'periphery_unit'])
    gdfs_to_merge.append(gdf_units_reduced)

    # LSOA catchment.
    # Pick out catchment areas of units that have been selected
    gdf_catchment_reduced = filter_gdf(gdf_boundaries_catchment, ['selected'])
    gdfs_to_merge.append(gdf_catchment_reduced)

    if gdf_lines_transfer is None:
        pass
    else:
        # Transfer lines.
        # Pick out catchment areas of units that have been selected
        gdf_lines_reduced = filter_gdf(gdf_lines_transfer, ['selected'])
        gdfs_to_merge.append(gdf_lines_reduced)

    # Combine all of the reduced gdf:
    gdf_combo = pd.concat(gdfs_to_merge, axis='rows', ignore_index=True)

    # Convert from DataFrame to GeoDataFrame:
    gdf_combo = geopandas.GeoDataFrame(
        gdf_combo,
        geometry='geometry',
        crs=gdfs_to_merge[0].crs
        )

    box, map_extent = get_selected_area_extent(gdf_combo, leeway)
    return box, map_extent


def get_selected_area_extent(
        gdf_selected,
        leeway=10000,
        ):
    """
    What is the spatial extent of everything in this GeoDataFrame?

    Inputs
    ------
    gdf_selected - GeoDataFrame.
    leeway       - float. Padding space around the edge of
                   the plots from the outermost thing of
                   interest. Units to match gdf units.

    Returns
    -------
    box        - polygon. The box we'll crop gdfs to.
    map_extent - list. [minx, maxx, miny, maxy] axis coordinates.
    """
    minx, miny, maxx, maxy = gdf_selected.geometry.total_bounds
    # Give this some leeway:
    minx -= leeway
    miny -= leeway
    maxx += leeway
    maxy += leeway
    map_extent = [minx, maxx, miny, maxy]
    # Turn the points into a box:
    box = Polygon((
        (minx, miny),
        (minx, maxy),
        (maxx, maxy),
        (maxx, miny),
        (minx, miny),
    ))
    return box, map_extent


def _keep_only_geometry_in_box(gdf, box):
    """
    Keep only rows of this gdf that intersect the box.

    If a region is partly in and partly outside the box,
    it will be included in the output gdf.

    Inputs
    ------
    gdf - GeoDataFrame.
    box - polygon.

    Returns
    -------
    gdf - GeoDataFrame. Input data reduced to only rows that
          intersect the box.
    """
    mask = gdf.geometry.intersects(box)
    gdf = gdf[mask]
    return gdf


def _restrict_geometry_edges_to_box(gdf, box):
    """
    Clip polygons to the given box.

    Inputs
    ------
    gdf - GeoDataFrame.
    box - polygon.

    Returns
    -------
    gdf - GeoDataFrame. Same as the input gdf but cropped so nothing
          outside the box exists.
    """
    gdf.geometry = gdf.geometry.intersection(box)
    return gdf


def _assign_label_coords_to_regions(gdf, col_point_label):
    """
    Assign coordinates for labels of region short codes.

    Inputs
    ------
    gdf             - GeoDataFrame.
    col_point_label - name of the column to place coords in.

    Returns
    -------
    gdf - GeoDataFrame. Same as input but with added coordinates.
    """
    # Get coordinates for where to plot each label:
    point_label = ([poly.representative_point() for
                    poly in gdf.geometry])
    gdf[col_point_label] = point_label
    return gdf


# ###########################
# ##### SETUP FOR PLOTS #####
# ###########################
def _setup_plot_map_outcome(
        gdf_boundaries_lsoa,
        scenario: str,
        outcome: str,
        boundary_kwargs={},
        ):
    """
    Pick out colour scale properties for outcome map.

    The required colour limits depend on the scenario type.
    If it's a 'diff' scenario, we want to take shared colour limits
    from only the other 'diff' scenarios and then use a symmetrical
    colour map and limits. Otherwise we want to exclude the 'diff'
    scenarios when finding shared colour limits.

    Inputs
    ------
    gdf_boundaries_lsoa - GeoDataFrame. Contains outcomes by LSOA.
    scenario            - str. Name of the scenario to be plotted.
    outcome             - str. Name of the outcome measure to be plotted.
    boundary_kwargs     - dict. Kwargs for the LSOA boundary plot
                          call later.

    Returns
    -------
    lsoa_boundary_kwargs - dict. Kwargs for the LSOA boundary plot
                           call later including colour scale limits.
    """
    # Find shared outcome limits.
    # Take only scenarios containing 'diff':
    mask = gdf_boundaries_lsoa.columns.get_level_values(
        'scenario').str.contains('diff')
    if scenario.startswith('diff'):
        pass
    else:
        # Take the opposite condition, take only scenarios
        # not containing 'diff'.
        mask = ~mask

    mask = (
        mask &
        (gdf_boundaries_lsoa.columns.get_level_values('subtype') == 'mean') &
        (gdf_boundaries_lsoa.columns.get_level_values('property') == outcome)
    )
    all_mean_vals = gdf_boundaries_lsoa.iloc[:, mask]
    vlim_abs = all_mean_vals.abs().max().values[0]
    vmax = all_mean_vals.max().values[0]
    vmin = all_mean_vals.min().values[0]

    lsoa_boundary_kwargs = {
        'column': (outcome, 'mean'),
        'edgecolor': 'face',
        # Adjust size of colourmap key, and add label
        'legend_kwds': {
            'shrink': 0.5,
            'label': outcome
            },
        # Set to display legend
        'legend': True,
        }

    cbar_diff = True if scenario.startswith('diff') else False
    if cbar_diff:
        lsoa_boundary_kwargs['cmap'] = 'seismic'
        lsoa_boundary_kwargs['vmin'] = -vlim_abs
        lsoa_boundary_kwargs['vmax'] = vlim_abs
    else:
        cmap = 'plasma'
        if outcome == 'mRS shift':
            # Reverse the colourmap because lower values
            # are better for this outcome.
            cmap += '_r'
        lsoa_boundary_kwargs['cmap'] = cmap
        lsoa_boundary_kwargs['vmin'] = vmin
        lsoa_boundary_kwargs['vmax'] = vmax
    # Update this with anything from the input dict:
    lsoa_boundary_kwargs = lsoa_boundary_kwargs | boundary_kwargs

    return lsoa_boundary_kwargs


def make_colours_for_catchment(
        gdf_boundaries_catchment,
        colour_lists_units=[],
        colour_list_periphery_units=[],
        col_colour_ind='colour_ind',
        col_transfer_colour_ind='transfer_colour_ind',
        col_selected='selected',
        col_use='use',
        col_output_colour='colour',
        col_output_colour_lines='colour_lines',
        col_output_colour_periphery='colour_periphery'
        ):
    """
    Assign colours to the catchment areas.

    If the transfer unit information exists in gdf_boundaries_catchment
    then colours will be chosen so that units sharing a transfer unit
    will share a base colour and be assigned different shades.
    Otherwise the same base colour is used for all catchment areas.

    Inputs
    ------
    gdf_boundaries_catchment    - GeoDataFrame. Contains colour index.
    colour_lists_units          - list. Cmaps or #rrggbbaa to be assigned.
    colour_list_periphery_units - list. Cmaps or #rrggbbaa to be assigned
                                  for periphery unit catchment areas.
    col_colour_ind              - str / tuple. Column of colour index.
    col_transfer_colour_ind     - str / tuple. Column of transfer unit
                                  colour index.
    col_selected                - str / tuple. Column of selected unit.
    col_use                     - str / tuple. Column of "use" for this
                                  scenario (different from "selected").
    col_output_colour           - str / tuple. Column of resulting colour.
    col_output_colour_lines     - str / tuple. Column of resulting colour
                                  for other uses, e.g. transfer unit lines.
    col_output_colour_periphery - str / tuple. Column of resulting colour
                                  for periphery units.

    Returns
    -------
    gdf_boundaries_catchment - GeoDataFrame. Same as input with added
                               columns with colours.
    """
    def make_colour_list(cmap='Blues', inds_cmap=[0.2, 0.4, 0.6, 0.8]):
        # Pick out colours from the cmap.
        colour_list = plt.get_cmap(cmap)(inds_cmap)
        # Convert from (0.0 to 1.0) to (0 to 255):
        colour_list = (colour_list * 255.0).astype(int)
        # Convert to string:
        colour_list = np.array([
            '#%02x%02x%02x%02x' % tuple(c) for c in colour_list])
        return colour_list

    # Pick sequential colourmaps that go from light (0.0) to dark (1.0).
    # Don't include Greys because it's used for periphery units.
    cmaps_list = [
        'Blues', 'Greens', 'Oranges', 'Purples', 'Reds',
        'YlOrBr', 'RdPu', 'BuGn', ' YlOrRd', 'BuPu',
    ]
    if len(colour_lists_units) == 0:
        colour_lists_units = cmaps_list

    mask = (gdf_boundaries_catchment[col_use] == 1)
    n_colours = len(list(set(
        gdf_boundaries_catchment.loc[mask, col_colour_ind])))

    # Selected units.
    # Start with blank colours:
    colours_units = np.array(['#00000000'] * len(gdf_boundaries_catchment))
    colours_transfer = np.array(['#00000000'] * len(gdf_boundaries_catchment))
    colours_periphery = np.array(['#00000000'] * len(gdf_boundaries_catchment))

    if col_transfer_colour_ind in gdf_boundaries_catchment.columns:
        # Use a different colour map for each MT unit and its feeders.
        gdf_boundaries_catchment = gdf_boundaries_catchment.copy()
        gdf_boundaries_catchment = gdf_boundaries_catchment.sort_values(
            col_selected, ascending=False)
        # Mask to basically select "use" only, but also the periphery
        # units have NaN instead of an integer value in this column
        # so use isna() to exclude those.
        mask = ~gdf_boundaries_catchment[col_transfer_colour_ind].isna()
        bands = list(gdf_boundaries_catchment.loc[
            mask, col_transfer_colour_ind].dropna().unique())

        for b, band in enumerate(bands):
            try:
                cmap = colour_lists_units[b]
                colour_list_units = make_colour_list(
                    cmap=cmap, inds_cmap=np.linspace(0.2, 0.8, n_colours))
                # Pick out a dark colour for transfer unit line:
                colour_transfer = make_colour_list(
                    cmap=cmap, inds_cmap=[0.95])[0]
            except ValueError:
                # No colourmap of that name.
                colour_list_units = colour_lists_units[b][:-1]
                colour_transfer = colour_lists_units[b][-1]
            # Pick out only the areas in this band:
            mask = (
                (gdf_boundaries_catchment[col_transfer_colour_ind] == band) &
                (gdf_boundaries_catchment[col_use] == 1)
            )
            # Assign colours by colour index column values:
            inds = gdf_boundaries_catchment.loc[
                mask, col_colour_ind].astype(int).values
            colours_units[np.where(mask == 1)[0]] = colour_list_units[inds]
            colours_transfer[np.where(mask == 1)[0]] = colour_transfer
    else:
        try:
            cmap = colour_lists_units[0]
            colour_list_units = make_colour_list(
                cmap=cmap, inds_cmap=np.linspace(0.2, 0.8, n_colours))
            # Pick out a dark colour for transfer unit line:
            colour_transfer = make_colour_list(
                cmap=cmap, inds_cmap=[0.95])[0]
        except ValueError:
            # No colourmap of that name.
            colour_list_units = colour_lists_units[b][:-1]
            colour_transfer = colour_lists_units[b][-1]
        # Assign colours by colour index column values:
        colours_units = colour_list_units[
            gdf_boundaries_catchment[col_colour_ind].astype(int).values]
        colours_transfer = np.array(
            [colour_transfer] * len(gdf_boundaries_catchment))

    # Place in the GeoDataFrame:
    gdf_boundaries_catchment[col_output_colour] = colours_units
    gdf_boundaries_catchment[col_output_colour_lines] = colours_transfer

    # Periphery units
    if len(colour_list_periphery_units) == 0:
        colour_list_periphery_units = make_colour_list(
            cmap='Greys', inds_cmap=np.linspace(0.2, 0.8, n_colours))
    else:
        pass
    mask = ~gdf_boundaries_catchment[col_colour_ind].isna()
    # Assign colours by colour index column values:
    colours_periphery[np.where(mask == 1)[0]] = colour_list_periphery_units[
        gdf_boundaries_catchment.loc[mask, col_colour_ind].astype(int).values]
    # Place in the GeoDataFrame:
    gdf_boundaries_catchment[col_output_colour_periphery] = colours_periphery
    return gdf_boundaries_catchment


def drop_other_scenarios(df, scenario):
    """
    Remove other scenarios and the 'scenario' column heading from df.

    Inputs
    ------
    df       - GeoDataFrame. Contains a column MultiIndex with a
               'scenario' level.
    scenario - str. Name of the scenario to keep.

    Returns
    -------
    df_selected - GeoDataFrame. Subset of the input dataframe with
                  only the selected scenario and the 'scenario' level
                  removed.
    """
    scenario_list = list(set(df.columns.get_level_values('scenario')))
    scenarios_to_keep = [s for s in scenario_list if (
        (s == scenario) |
        (s == '') |
        (s == 'any') |
        (s.startswith('Unnamed'))
    )]

    # Which columns do we want?
    cols = df.columns.get_level_values('scenario').isin(scenarios_to_keep)
    # Subset of only these columns:
    df_selected = df.loc[:, cols].copy()

    # Drop the 'scenario' column level:
    df_selected = df_selected.droplevel('scenario', axis='columns')
    # Set geometry:
    col_geometry = find_multiindex_column_names(
        df_selected, property=['geometry'])
    df_selected = df_selected.set_geometry(col_geometry)

    return df_selected


# #######################
# ##### PYPLOT MAPS #####
# #######################
def plot_map_selected_regions(
        gdf_boundaries_regions,
        gdf_points_units,
        scenario: str,
        map_extent=[],
        path_to_file='',
        show=True
        ):
    """
    Plot a map labelling selected and nearby regions and units.

    +---+ +---------+ +---+    1 - Legend for units, selected / other.
    | 1 | |    2    | | 3 |    2 - Map with regions and units.
    +---+ +---------+ +---+    3 - Legend for regions, selected / other.

    Inputs
    ------
    gdf_boundaries_regions - GeoDataFrame.
    gdf_points_units       - GeoDataFrame.
    scenario               - str. Name of scenario to use.
    map_extent             - list. Axis limits [xmin, xmax, ymin, ymax].
    path_to_file           - str. Save the image here.
    show                   - bool. Whether to plt.show().
    """
    # Drop everything that belongs to other scenarios:
    gdf_boundaries_regions = drop_other_scenarios(
        gdf_boundaries_regions, scenario)
    gdf_points_units = drop_other_scenarios(
        gdf_points_units, scenario)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax, extra_artists = maps.plot_map_selected_regions(
        gdf_boundaries_regions,
        gdf_points_units,
        ax=ax,
        map_extent=map_extent
    )

    if path_to_file is None:
        pass
    else:
        # Include extra artists so that bbox_inches='tight' line
        # in savefig() doesn't cut off the legends.
        # Adding legends with ax.add_artist() means that the
        # bbox_inches='tight' line ignores them.
        plt.savefig(
            path_to_file,
            bbox_extra_artists=extra_artists,
            dpi=300, bbox_inches='tight'
            )
    if show:
        # Add dummy axis to the sides so that
        # extra_artists are not cut off when plt.show() crops.
        fig = maps.plot_dummy_axis(fig, ax, extra_artists[1], side='left')
        fig = maps.plot_dummy_axis(fig, ax, extra_artists[0], side='right')
        plt.show()
    else:
        plt.close()


def plot_map_selected_units(
        gdf_boundaries_regions,
        gdf_points_units,
        gdf_lines_transfer,
        scenario: str,
        map_extent=[],
        path_to_file='',
        show=True
        ):
    """
    Plot a map labelling selected and nearby units and transfers.

    +---+ +---------+    1 - Legend for units, selected / other.
    | 1 | |    2    |    2 - Map with regions, units, transfer lines.
    +---+ +---------+

    Inputs
    ------
    gdf_boundaries_regions - GeoDataFrame.
    gdf_points_units       - GeoDataFrame.
    gdf_lines_transfer     - GeoDataFrame.
    scenario               - str. Name of scenario to use.
    map_extent             - list. Axis limits [xmin, xmax, ymin, ymax].
    path_to_file           - str. Save the image here.
    show                   - bool. Whether to plt.show().
    """
    # Drop everything that belongs to other scenarios:
    gdf_boundaries_regions = drop_other_scenarios(
        gdf_boundaries_regions, scenario)
    gdf_points_units = drop_other_scenarios(gdf_points_units, scenario)
    gdf_lines_transfer = drop_other_scenarios(gdf_lines_transfer, scenario)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax, extra_artists = maps.plot_map_selected_units(
        gdf_boundaries_regions,
        gdf_points_units,
        gdf_lines_transfer,
        ax=ax,
        map_extent=map_extent,
    )

    if path_to_file is None:
        pass
    else:
        # Include extra artists so that bbox_inches='tight' line
        # in savefig() doesn't cut off the legends.
        # Adding legends with ax.add_artist() means that the
        # bbox_inches='tight' line ignores them.
        plt.savefig(
            path_to_file,
            bbox_extra_artists=extra_artists,
            dpi=300, bbox_inches='tight'
            )
    if show:
        # Add dummy axis to the sides so that
        # extra_artists are not cut off when plt.show() crops.
        fig = maps.plot_dummy_axis(fig, ax, extra_artists[0], side='left')
        plt.show()
    else:
        plt.close()


def plot_map_catchment(
        gdf_boundaries_catchment,
        gdf_boundaries_regions,
        gdf_points_units,
        gdf_lines_transfer,
        scenario: str,
        title='',
        lsoa_boundary_kwargs={},
        lsoa_boundary_periphery_kwargs={},
        map_extent=[],
        show=True,
        path_to_file=''
        ):
    """
    Plot a map of unit catchment areas.

    +---+ +---------+    1 - Legend for units, selected / other.
    | 1 | |    2    |    2 - Map with regions, units, transfer lines,
    +---+ +---------+        catchment areas.

    Inputs
    ------
    gdf_boundaries_catchment - GeoDataFrame.
    gdf_boundaries_regions   - GeoDataFrame.
    gdf_points_units         - GeoDataFrame.
    gdf_lines_transfer       - GeoDataFrame.
    scenario                 - str. Name of scenario to use.
    title                    - str. Title of the axis.
    lsoa_boundary_kwargs     - dict. Kwargs for plotting catchment areas.
    lsoa_boundary_periphery_kwargs - dict. Kwargs for periphery areas.
    map_extent               - list. Axis limits [xmin, xmax, ymin, ymax].
    path_to_file             - str. Save the image here.
    show                     - bool. Whether to plt.show().
    """
    # Drop everything that belongs to other scenarios:
    gdf_boundaries_regions = drop_other_scenarios(
        gdf_boundaries_regions, scenario)
    gdf_points_units = drop_other_scenarios(gdf_points_units, scenario)
    gdf_lines_transfer = drop_other_scenarios(gdf_lines_transfer, scenario)
    gdf_boundaries_catchment = drop_other_scenarios(
        gdf_boundaries_catchment, scenario)
    # Drop catchment areas from other scenarios:
    gdf_boundaries_catchment = gdf_boundaries_catchment.dropna(
        subset='colour_ind')

    boundary_kwargs = {'edgecolor': 'face'}
    # Update this with anything from the input dict:
    lsoa_boundary_kwargs = boundary_kwargs | lsoa_boundary_kwargs

    boundary_periphery_kwargs = {'edgecolor': 'face'}
    # Update this with anything from the input dict:
    lsoa_boundary_periphery_kwargs = (boundary_periphery_kwargs |
                                      lsoa_boundary_periphery_kwargs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title(title)

    ax, extra_artists = maps.plot_map_catchment(
        gdf_boundaries_catchment,
        gdf_boundaries_regions,
        gdf_points_units,
        gdf_lines_transfer,
        ax=ax,
        map_extent=map_extent,
        boundary_kwargs=lsoa_boundary_kwargs,
        boundary_periphery_kwargs=lsoa_boundary_periphery_kwargs
    )

    if path_to_file is None:
        pass
    else:
        # Include extra artists so that bbox_inches='tight' line
        # in savefig() doesn't cut off the legends.
        # Adding legends with ax.add_artist() means that the
        # bbox_inches='tight' line ignores them.
        plt.savefig(
            path_to_file,
            bbox_extra_artists=extra_artists,
            dpi=300, bbox_inches='tight')
    if show:
        # Add dummy axis to the sides so that
        # extra_artists are not cut off when plt.show() crops.
        fig = maps.plot_dummy_axis(fig, ax, extra_artists[0], side='left')
        plt.show()
    else:
        plt.close()


def plot_map_outcome(
        gdf_boundaries_lsoa,
        gdf_boundaries_regions,
        gdf_points_units,
        scenario,
        outcome,
        title='',
        lsoa_boundary_kwargs={},
        map_extent=[],
        draw_region_boundaries=True,
        show=True,
        path_to_file=None
        ):
    """
    Plot a map of LSOA outcomes.

    +---+ +---------+ |     1 - Legend for units, selected / other.
    | 1 | |    2    | | 3   2 - Map with regions, units, LSOA by outcome.
    +---+ +---------+ |     3 - Colourbar for outcome.

    Inputs
    ------
    gdf_boundaries_catchment - GeoDataFrame.
    gdf_boundaries_regions   - GeoDataFrame.
    gdf_points_units         - GeoDataFrame.
    scenario                 - str. Name of scenario to use.
    outcome                  - str. Name of outcome to show.
    title                    - str. Title of the axis.
    lsoa_boundary_kwargs     - dict. Kwargs for plotting catchment areas.
    map_extent               - list. Axis limits [xmin, xmax, ymin, ymax].
    draw_region_boundaries   - bool. Whether to draw regions in background.
    path_to_file             - str. Save the image here.
    show                     - bool. Whether to plt.show().
    """
    lsoa_boundary_kwargs = _setup_plot_map_outcome(
        gdf_boundaries_lsoa,
        scenario,
        outcome,
        boundary_kwargs=lsoa_boundary_kwargs,
        )

    # Drop everything that belongs to other scenarios:
    gdf_boundaries_regions = drop_other_scenarios(
        gdf_boundaries_regions, scenario)
    gdf_points_units = drop_other_scenarios(gdf_points_units, scenario)
    gdf_boundaries_lsoa = drop_other_scenarios(gdf_boundaries_lsoa, scenario)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(title)

    ax, extra_artists = maps.plot_map_outcome(
        gdf_boundaries_lsoa,
        gdf_boundaries_regions,
        gdf_points_units,
        ax=ax,
        map_extent=map_extent,
        boundary_kwargs=lsoa_boundary_kwargs,
        draw_region_boundaries=draw_region_boundaries,
    )

    if path_to_file is None:
        pass
    else:
        # Include extra artists so that bbox_inches='tight' line
        # in savefig() doesn't cut off the legends.
        # Adding legends with ax.add_artist() means that the
        # bbox_inches='tight' line ignores them.
        plt.savefig(
            path_to_file,
            bbox_extra_artists=extra_artists,
            dpi=300, bbox_inches='tight'
            )
    if show:
        # Add dummy axis to the sides so that
        # extra_artists are not cut off when plt.show() crops.
        fig = maps.plot_dummy_axis(fig, ax, extra_artists[0], side='left')
        plt.show()
    else:
        plt.close()
