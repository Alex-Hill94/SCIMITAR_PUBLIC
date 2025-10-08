import vtk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, lines, colors, cm
import interactor_utils
from aux_material import *
from tqdm import tqdm
import pandas as pd

# Suppress VTK intersection warnings 
vtk.vtkObject.GlobalWarningDisplayOff()

class Scimitar():

    ''' 
    A class to create, visualise and study a hypothetical X-Ray scanner setup. The X-ray scanner scene is comprised of panels, emitters, a patient, and a detector. Each panel contains an array of emitters, which illuminate the patient lying on the detector surface. 
    
    Inputs:
        panel_pitch (float [metres]):   The horizontal and vertical distance separating the centres of a neighbouring pair of panels.
        cone_angle  (float [degrees]):  The maximum opening angle of the cone.
        panel_theta (float [degrees]):  The angle the each panel is rotated with respect to its unique axis of rotation.
        activate    (int array):        An array that controls which of the nine default panels are included in this scene. The array is ordered [top_left, top, top_right, left, centre, right, bottom_left, bottom, bottom_right], and 1 = on, 0 = off.
        cone_resolution (int):          The number of vertices describing the cross section of the cone.
        SID         (float [metres]):            The source-to-image distance, the height between the centre of the central panel and the detector surface.
        grid        (int):              Emitters are by default arranged in a square grid. This argument determines the number of emitters on a side, meaning that there are grid^2 emitters per panel.

    Outputs:
        emitters_master_list (list [vtk objects]):  A list of all the emitters in the scene. The length of this array will be grid^2 * sum(activate).
        plane_probe_master_list (list [vtk objects]):  A list of all the intersections between the emitters and the detector in the scene. 
        patient_probe_master_list (list [vtk objects]):  A list of all the intersections between the emitters and the patient in the scene. 
        
        PlanePoints (float array [metres]):     The coordinates of the vertices of all the intersections between emitters and the plane.
        PlaneTris (int array [metres]):         The triangles describing all the intersections between emitters and the plane.
        PlaneScalars (float array [metres]):    The scalar associated with each vertex of all the intersections between emitters and the plane.
        
    '''

    def __init__(self):
        
        self.panel_pitch = []
        self.cone_angle = []
        self.panel_theta = []
        self.panel_alpha = []
        self.cone_alpha = []
        self.intersection_heights = []
        self.activate = []
        self.panel_setting = []
        self.panel_positions = []
        self.stagger = []
        self.cone_resolution = []
        self.SID = []
        self.intersection_heights = []
        self.grid = []

        self.emitters_master_list = []
        self.plane_probe_master_list = []
        self.patient_probe_master_list = []
        self.PlanePoints = []
        self.PlaneTris = []
        self.PlaneScalars = []

    def _process_panel(self, i, panel_actors, emitter_actors, intersection_heights,
                    cone_resolution, PlanePoints, PlaneTris, PlaneScalars):
        """
        Compute cone-plane intersections for a single detector panel.

        Returns
        -------
        emitters_temp_list, plane_probe_temp_list, patient_probe_temp_list : list
            Temporary lists of emitters and intersection plane actors.
        """
        emitters_temp_list = []
        plane_probe_temp_list = []
        patient_probe_temp_list = []

        n_em_per_pan = len(emitter_actors[0])
        n_planes = len(intersection_heights)

        for j in tqdm(range(n_em_per_pan), disable=self.silence_tqdm):
            index = i * n_em_per_pan + j
            emitter = emitter_actors[i][j]
            emitters_temp_list.append(emitter)

            for k in range(n_planes):
                plane_height = intersection_heights[k]
                plane = Plane(height=plane_height)

                plane_intersection = clipping_utils.clipper(
                    emitter,
                    clipping_function=plane,
                    fill=True,
                    opacity=1,
                    name=f"panel-{i}/intersections/detector-intersection-{j}",
                )

                plane_points, plane_tris = clipping_utils.extract_triangles(plane_intersection)
                plane_scalars = np.zeros(plane_points.shape[0])

                if k == 0:
                    plane_probe_temp_list.append(plane_intersection)

                PlanePoints[k][index] = plane_points
                PlaneTris[k][index] = plane_tris
                PlaneScalars[k][index] = plane_scalars

        return emitters_temp_list, plane_probe_temp_list, patient_probe_temp_list

    def _generate_scene_geometry(self, panel_array_obj, intersection_heights, cone_resolution):
        """
        Internal helper to build cone-plane intersection geometry, 
        store visualisation actors, and populate geometric data arrays.

        Parameters
        ----------
        panel_array_obj : PanelArray
            Constructed panel array object containing panel and emitter actors.
        intersection_heights : list of float
            Z-heights of intersection planes.
        cone_resolution : int
            Number of polygonal segments used to approximate cone geometry.
        """

        # --- Retrieve configuration data ---
        panel_actors = panel_array_obj.panel_actors
        emitter_actors = panel_array_obj.emitter_actors
        cone_positions = panel_array_obj.cone_positions

        self.centred_emitters = panel_array_obj.centred_emitters
        self.wafer_size = panel_array_obj.wafer_size

        # --- Validate geometry ---
        n_emitters = len(np.ravel(emitter_actors))
        n_panels = len(panel_actors)
        n_em_per_pan = len(emitter_actors[0])
        n_planes = len(intersection_heights)

        assert n_emitters == n_panels * n_em_per_pan, (
            f"Unexpected number of emitters! Got {n_emitters}, expected {n_panels * n_em_per_pan}"
        )

        # --- Initialise master storage structures ---
        emitters_master_list = []
        plane_probe_master_list = []
        patient_probe_master_list = []

        PlanePoints = nan_array((n_planes, n_emitters, cone_resolution, 3))
        PlaneTris = nan_array((n_planes, n_emitters, cone_resolution - 2, 3))
        PlaneScalars = nan_array((n_planes, n_emitters, cone_resolution))

        # --- Build intersections for each panel ---
        for i in tqdm(range(n_panels), desc="Adding Panels", disable=self.silence_tqdm):
            emitters_temp, plane_probe_temp, patient_probe_temp = self._process_panel(
                i, panel_actors, emitter_actors, intersection_heights,
                cone_resolution, PlanePoints, PlaneTris, PlaneScalars
            )
            emitters_master_list.append(emitters_temp)
            plane_probe_master_list.append(plane_probe_temp)
            patient_probe_master_list.append(patient_probe_temp)

        # --- Assign results to self ---
        self.emitters_master_list = emitters_master_list
        self.plane_probe_master_list = plane_probe_master_list
        self.patient_probe_master_list = patient_probe_master_list
        self.PlanePoints = PlanePoints
        self.PlaneTris = PlaneTris
        self.PlaneScalars = PlaneScalars
        self.panel_actors = panel_actors
        self.emitter_actors = emitter_actors
        self.cone_positions = cone_positions
        self.additional_actors = []

    def _point_in_triangle(self, p, v1, v2, v3):
        """Checks whether a point is within the given triangle

        The function checks, whether the given point p is within the triangle defined by the the three corner point v1,
        v2 and v3.
        This is done by checking whether the point is on all three half-planes defined by the three edges of the triangle.
        :param p: The point to be checked (tuple with x any y coordinate)
        :param v1: First vertex of the triangle (tuple with x any y coordinate)
        :param v2: Second vertex of the triangle (tuple with x any y coordinate)
        :param v3: Third vertex of the triangle (tuple with x any y coordinate)
        :return: True if the point is within the triangle, False if not
        """
        def _test(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        b1 = _test(p, v1, v2) < 0.0
        b2 = _test(p, v2, v3) < 0.0
        b3 = _test(p, v3, v1) < 0.0
        return (b1 == b2) * (b2 == b3) 

    def _irradiation_at_all_heights(self, pix_per_cm=2, xy_lims=[-1.0, 1.0]):
        """
        Compute 2D irradiation maps for all detector planes.

        This method calculates per-emitter, per-panel, and total irradiation
        distributions across all intersection planes in the current scanner
        geometry. Each map is generated by evaluating which pixels fall within
        the triangular projections of X-ray emitter cones.

        Parameters
        ----------
        pix_per_cm : int, optional
            Spatial resolution of the detector grid (pixels per cm). Default is 2.
        xy_lims : list of float, optional
            The [min, max] coordinate limits (in cm) defining the square detector
            area. Default is [-1.0, 1.0].

        Returns
        -------
        None
            Updates the instance with computed irradiation layers and metadata:
            `irradiation_layers`, `irradiation_layers_e`, `irradiation_layers_p`,
            `x_range`, `y_range`, `vmax`, and pixel geometry attributes.
        """
        # --- Geometry setup ---
        PlanesPoints = self.PlanePoints
        PlanesTris = self.PlaneTris
        PlanesHeight = self.intersection_heights
        n_panels = len(self.panel_actors)
        n_planes = len(PlanesPoints)
        n_emitters = len(PlanesPoints[0])

        # --- 2D pixel grid setup ---
        xy_lims = np.array(xy_lims)
        pix_length = int(round(1e2 * (xy_lims.ptp()) * pix_per_cm, 3))
        x_edges = np.linspace(xy_lims[0], xy_lims[1], pix_length + 1)
        y_edges = np.linspace(xy_lims[0], xy_lims[1], pix_length + 1)
        x_range = x_edges[:-1] + np.diff(x_edges) / 2
        y_range = y_edges[:-1] + np.diff(y_edges) / 2
        X2D, Y2D = np.meshgrid(x_range, y_range)
        xs, ys = X2D.ravel(), Y2D.ravel()
        vmax = np.sum(self.activate) * self.grid**2

        # --- Preallocate output arrays ---
        irradiation_layers   = np.zeros((n_planes, pix_length, pix_length))
        irradiation_layers_e = np.zeros((n_planes, n_emitters, pix_length, pix_length))
        irradiation_layers_p = np.zeros((n_planes, n_panels, pix_length, pix_length))

        # --- Local helper functions ---
        def _get_tri_coords(tris, points):
            """Return coordinates of triangle vertices."""
            tris = tris.astype(int)
            return np.array([[points[a], points[b], points[c]] for a, b, c in tris])

        def _get_minmax(plane_points):
            """Return bounding box across all emitters for one plane."""
            xs_min = [np.min(p[:, 0]) for p in plane_points]
            xs_max = [np.max(p[:, 0]) for p in plane_points]
            ys_min = [np.min(p[:, 1]) for p in plane_points]
            ys_max = [np.max(p[:, 1]) for p in plane_points]
            return np.min(xs_min), np.max(xs_max), np.min(ys_min), np.max(ys_max)

        def _irradiation_map(plane_idx):
            """Compute irradiation map for one plane."""
            PlanePoints = PlanesPoints[plane_idx]
            PlaneTris = PlanesTris[plane_idx]

            # Bounding box limits for pixel masking
            x_min, x_max, y_min, y_max = _get_minmax(PlanePoints)
            mask = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)
            xs_m, ys_m = xs[mask], ys[mask]

            # Initialise maps
            total = np.zeros(xs.shape)
            per_emitter = np.zeros((n_emitters, xs.shape[0]))

            for e_idx in tqdm(range(n_emitters), disable=self.silence_tqdm):
                tris_coords = _get_tri_coords(PlaneTris[e_idx], PlanePoints[e_idx])
                emitter_map = np.zeros(xs_m.shape)
                for v1, v2, v3 in tris_coords:
                    emitter_map += self._point_in_triangle(np.vstack((xs_m, ys_m)), v1, v2, v3)
                emitter_map = np.clip(emitter_map, 0, 1)
                total[mask] += emitter_map
                per_emitter[e_idx, mask] = emitter_map

            # Reshape to 2D images
            ir_map   = total.reshape(pix_length, pix_length)
            ir_map_e = per_emitter.reshape(n_emitters, pix_length, pix_length)
            ir_map_p = ir_map_e.reshape(n_panels, n_emitters // n_panels, pix_length, pix_length).sum(axis=1)
            return ir_map, ir_map_e, ir_map_p

        # --- Main loop over detector planes ---
        for k in range(n_planes):
            ir_map, ir_map_e, ir_map_p = _irradiation_map(k)
            irradiation_layers[k] = ir_map
            irradiation_layers_e[k] = ir_map_e
            irradiation_layers_p[k] = ir_map_p

        # --- Store results ---
        self.irradiation_layers = irradiation_layers
        self.irradiation_layers_e = irradiation_layers_e
        self.irradiation_layers_p = irradiation_layers_p
        self.x_range, self.y_range = x_range, y_range
        self.x_range_edges, self.y_range_edges = x_edges, y_edges
        self.vmax = vmax
        self.pix_length = pix_length
        self.pix_per_cm = pix_per_cm
    
    def Scene(  self,
                grid = 5,
                SID = 0.6,
                panel_pitch = 0.6,
                emitter_pitch = 0.025,
                cone_angle = 20,
                panel_theta = 45,
                panel_alpha = 0,
                cone_alpha = 45, 
                intersection_heights = [0.],
                activate = all_panels,
                panel_setting = 'standard grid',
                panel_positions = None,
                stagger = None,
                cone_resolution = 4,
                force_vis = False,
                silence_tqdm = True):

        """
        Construct and initialise the full 3D X-ray DT scanner scene, including panel arrays, emitters,
        and cone-beam intersections with defined planes for irradiation analysis or visualisation.

        This method generates all necessary VTK objects (panels, emitters, cones, intersection planes)
        describing the scanner geometry. It validates the physical configuration, such as checking for panel
        overlaps and X-ray cone collisions, and computes geometric intersection data used for
        subsequent irradiation and performance analysis.

        Parameters
        ----------
        grid : int, optional
            Number of emitters on a side of a square grid etched onto each flat panel. Default is 5.
        SID : float, optional
            Source-to-image distance (m), i.e. the normal distance between the bottom of the central panel and the detector surface. Default is 0.6.
        panel_pitch : float, optional
            Distance (m) between the centers of adjacent flat panels. Default is 0.6.
        emitter_pitch : float, optional
            Distance (m) between adjacent emitters on a single panel. Default is 0.025.
        cone_angle : float, optional
            Full cone angle (degrees) of X-ray emission for each emitter source. Default is 20.
        panel_theta : float, optional
            Tilt angle (degrees) of each flat panel relative to each panel's defined axis of rotation central axis. Default is 45.
        panel_alpha : float, optional
            In-plane rotation (degrees) of flat panels. Default is 0.
        cone_alpha : float, optional
            Rotation (degrees) of emission cones relative to the panel normal. Default is 45.
        intersection_heights : list of float, optional
            Z-heights (m) of planes used to compute cone intersections (e.g., detector or patient planes).
            `z = 0`, as the detector height, is automatically included if omitted. Default is [0.].
        activate : list or callable, optional
            Selection mask determining which panels are active. Default is `all_panels`. See aux_material.py for a full list.
        panel_setting : str, optional
            Layout type for the panel array (e.g., 'standard grid', 'custom'). Default is 'standard grid'.
        panel_positions : ndarray or None, optional
            Explicit (x, y, z) positions for panel centers if custom placement is specified. Default is None.
        stagger : float or None, optional
            Optional positional offset applied to central. Default is None. If activated, units are in metres (m).
        cone_resolution : int, optional
            Number of polygonal segments approximating each emission cone. Default is 4, i.e. a square cone.
        force_vis : bool, optional
            Force visualisation and intersection computation even if physical checks fail (e.g., collisions). Default is False.
        silence_tqdm : bool, optional
            Suppress progress bar output during construction. Default is True.

        Returns
        -------
        None
            The method populates several class attributes including:
            - `df` : pandas.DataFrame  
            Summary table of input parameters, computed overlaps, and collision metrics.
            - `panel_actors`, `emitter_actors` : list  
            VTK actor objects for visualisation of the physical geometry.
            - `PlanePoints`, `PlaneTris`, `PlaneScalars` : ndarray  
            Arrays storing geometric data of cone-plane intersections.
            - `panel_collision`, `panelXRay_collision`, `outside_wafer` : bool  
            Flags indicating physical realism of setup.

        Notes
        -----
        - Plane intersections are computed using `clipping_utils.clipper()`.
        - Intended for use within simulation workflows (e.g., SCIMITAR framework) to
        generate reproducible scanner configurations and geometric data for optimisation.

        """

        ### INPUTS
        self.panel_pitch = panel_pitch
        self.emitter_pitch = emitter_pitch
        self.cone_angle = cone_angle
        self.panel_theta = panel_theta
        self.panel_alpha = panel_alpha
        self.cone_alpha = cone_alpha
        self.activate = activate
        self.panel_setting = panel_setting
        self.panel_positions = panel_positions
        self.stagger = stagger
        self.cone_resolution = cone_resolution
        self.SID = SID
        self.grid = grid
        if np.isin(0.0, intersection_heights) == False:
            print('First intersection height is not at z = 0, inserting z = 0 value now.')
            intersection_heights = np.append(0.0, intersection_heights)
        intersection_heights = np.sort(intersection_heights)
        self.intersection_heights = intersection_heights
        self.force_vis = force_vis
        self.silence_tqdm = silence_tqdm
        self.wafer_size = 0.1
        

        input_data = [self.cone_angle, self.SID, self.emitter_pitch,
                        self.panel_pitch, self.panel_theta]
        
        ang_range = np.nan
        ilm       = np.nan
        o2d       = np.nan
        om2       = np.nan
        o3d       = np.nan
        stray     = np.nan

        output_data =  [ang_range,
                        ilm,
                        o2d,
                        om2,
                        o3d,
                        stray]

        ## Construct scanner object and assess whether physical relalism checks are met

        C = PanelArray()
        C.Construct(panel_pitch = panel_pitch, emitter_pitch = emitter_pitch, 
                    cone_angle = cone_angle, panel_theta = panel_theta, 
                    panel_alpha = panel_alpha, cone_alpha = cone_alpha, 
                    activate = activate, panel_setting = panel_setting,  
                    panel_positions = panel_positions, stagger = stagger, 
                    cone_resolution=cone_resolution,
                    SID = SID, grid = grid, 
                    silence_tqdm = silence_tqdm)

        pan_col = np.nan
        pn_xcol = np.nan
        out_waf = False
        acc_str   = np.nan
        con_pan = np.nan

        pan_col = C.panel_collision
        pn_xcol = C.panelXRay_collision
        out_waf = C.outside_wafer
        acc_str   = np.nan
        con_pan = C.contained_panel_dist

        checks = [  pan_col,
                    pn_xcol,
                    out_waf,
                    acc_str,
                    con_pan]

        column_names = [
                'Cone_Angle', 'SID', 'Emitter_Pitch', 'Panel_Pitch', 'Panel_Angle',
                'Angular_Range', 'Illumination', 'Overlap_2D', 'Overlap_Mid_2D', 'Overlap_3D',
                'Stray_Radiation', 'Panel_Collision', 'Panel_Xray_Collision', 'Outside_Wafer',
                'Acceptable_Stray', 'Contained_Panels'
            ]
    
        simulation_data =       np.vstack((input_data[0], input_data[1], input_data[2], 
                               input_data[3], input_data[4], 
                               output_data[0], output_data[1], output_data[2], 
                               output_data[3], output_data[4], output_data[5], 
                               checks[0], checks[1], int(checks[2]), 
                               checks[3], checks[4])).T

        # Initial pandas dataframe, to be returned if full computation is not completed

        df = pd.DataFrame(simulation_data, columns=column_names)
        
        self.df = df
        self.panel_collision = C.panel_collision
        self.panelXRay_collision = C.panelXRay_collision
        self.outside_wafer = C.outside_wafer
        self.contained_panel_dist = C.contained_panel_dist
        self.acceptable_stray_at_detector = np.nan

        if self.panel_collision + self.panelXRay_collision + self.outside_wafer == 0 or self.force_vis:
            self._generate_scene_geometry(C, intersection_heights, cone_resolution)

    def CheckStray(self,
                    d_width = 0.43,
                    box_centre = [0,0], 
                    layer = [0]):
        
        """
        Evaluate whether the modelled radiation strays significantly outside the 
        area of an assumed square detector with width 43cm.

        This method checks if the projected points on the detector plane 
        remain within a square box of width `d_width` centered at `box_centre`. 
        The degree of "stray" is quantified by how far the points extend 
        beyond the bounding box along the x- and y-directions. 
        A configuration is considered acceptable if the stray distances 
        remain within a threshold fraction of the source-to-image distance (SID).

        Parameters
        ----------
        d_width : float, optional
            Width of the bounding box (in detector-plane units). 
            Default is 0.43.
        box_centre : list of float, optional
            [x, y] coordinates of the bounding box centre. 
            Default is [0, 0].
        layer : list of int, optional
            Index of the irradiation layer to evaluate. 
            Default is [0], i.e. the detector layer.

        Notes
        -----
        - If panel collisions or invalid geometry are detected 
        (`self.panel_collision`, `self.panelXRay_collision`, or `self.outside_wafer` 
        are nonzero), the check is skipped and the result is forced to `False`.
        This is done to save time if a large set of inputs is being evaluated.

        Attributes
        ----------
        acceptable_stray_at_detector : bool
            True if the configuration satisfies stray tolerance criteria, 
            False otherwise.

        Returns
        -------
        None
            The result is stored in `self.acceptable_stray_at_detector`.
        """


        def _check_stray_at_layer(layer):
            DetectorPlanePoints = self.PlanePoints[layer]
            d = DetectorPlanePoints
            xs = np.ravel(d[:,:,0])
            ys = np.ravel(d[:,:,1])

            x_sl = box_x_extents[0] - xs
            x_sr = xs - box_x_extents[1]

            y_sd = box_y_extents[0] - ys 
            y_su = ys - box_y_extents[1]

            x_stray = [np.max(x_sl), np.max(x_sr)]
            x_stray = [0 if val < 0 else val for val in x_stray]

            y_stray = [np.max(y_sd), np.max(y_su)]
            y_stray = [0 if val < 0 else val for val in y_stray]


            condition_one = (np.sum(x_stray)) <= 0.03 * self.SID
            condition_two = (np.sum(y_stray)) <= 0.03 * self.SID
            condition_three = (np.sum(x_stray) + np.sum(y_stray)) <= 0.04 * self.SID
            acceptable_stray = condition_one * condition_two * condition_three
            return acceptable_stray

        if (self.panel_collision + self.panelXRay_collision + self.outside_wafer) != 0:
            if self.silence_tqdm == False:
                print('Intersections not computed due to invalid geometry, returning acceptable_stray_at_detector = False')
            acceptable_stray = False
        else:
            ### Extents of detector
            box_x_extents = [box_centre[0] - d_width/2, box_centre[0] + d_width/2]
            box_y_extents = [box_centre[1] - d_width/2, box_centre[1] + d_width/2]
            acceptable_stray = _check_stray_at_layer(layer[0])
        self.acceptable_stray_at_detector = acceptable_stray

    def Irradiation(self, 
                    pix_per_cm = 2, 
                    xy_lims = [-1., 1.]):
        """
        Compute the irradiation maps for the current X-ray scanner geometry.

        This method generates 2D irradiation distributions on detector planes by 
        evaluating which pixels fall inside the triangular regions defined by 
        projected X-ray emitter intersections with the detector. It constructs 
        irradiation maps at the emitter-, panel-, and total-detector levels. 
        The computed maps are stored as attributes for later visualisation and 
        analysis (see `Metrics` for plotting and further evaluation).

        Parameters
        ----------
        pix_per_cm : int, optional
            Number of pixels per centimeter for the discretisation of the 
            detector plane. Controls spatial resolution of the irradiation map. 
            Default is 2.
        xy_lims : list of float, optional
            The [min, max] extent (in cm) of the detector plane in both x and y 
            directions. Defines the square region over which irradiation maps 
            are calculated. Default is [-1.0, 1.0].

        Notes
        -----
        - Irradiation is only computed if the geometry is valid 
        (i.e., no panel collisions, no cross-irradiation, and emitters are 
        within wafer bounds).
        - The computation is performed separately for each detector plane, 
        emitter, and panel, allowing flexible analysis at different levels 
        of granularity.
        - For invalid geometries, no maps are generated.

        Attributes
        ----------
        irradiation_layers : ndarray of shape (n_planes, pix_length, pix_length)
            Total irradiation map per detector plane.
        irradiation_layers_e : ndarray of shape (n_planes, n_emitters, pix_length, pix_length)
            Irradiation contributions per emitter per plane.
        irradiation_layers_p : ndarray of shape (n_planes, n_panels, pix_length, pix_length)
            Irradiation contributions per panel per plane.
        x_range : ndarray
            Pixel center coordinates along the x-axis.
        y_range : ndarray
            Pixel center coordinates along the y-axis.
        x_range_edges : ndarray
            Pixel edge coordinates along the x-axis.
        y_range_edges : ndarray
            Pixel edge coordinates along the y-axis.
        vmax : float
            Maximum possible irradiation value (normalisation factor).
        pix_length : int
            Number of pixels along one axis of the detector plane.
        pix_per_cm : int
            Resolution used for the irradiation map (pixels per cm).

        Returns
        -------
        None
            The method updates the irradiation maps and related metadata 
            as attributes of the class instance.
        """


        if (self.panel_collision + self.panelXRay_collision + self.outside_wafer)>=1:
            if self.silence_tqdm == False:
                print('Invalid geometry, no irradiation maps created')
        else:    
            self._irradiation_at_all_heights(pix_per_cm, xy_lims)

    def Metrics(self, d_width=0.43):
        """
        Computes irradiation metrics for the chest DT device design.
        
        Parameters
        ----------
        d_width : float, optional
            Width of the square detector in metres, centered at (0,0) in the x-y plane.
            Default is 0.43.
            
        Returns
        -------
        None
            Results are stored in self.df as a pandas DataFrame with the following columns:
            - Cone_Angle: X-ray cone angle
            - SID: Source-to-image distance
            - Emitter_Pitch: Spacing between emitters
            - Panel_Pitch: Spacing between detector panels
            - Panel_Angle: Detector panel angle
            - Angular_Range: Maximum angular separation between emitter pairs
            - Illumination: Percentage of detector area covered by radiation
            - Overlap_2D: 2D overlap metric at detector surface
            - Overlap_Mid_2D: 2D overlap metric at mid-layer
            - Overlap_3D: 3D volumetric overlap metric
            - Stray_Radiation: Total stray radiation outside detector
            - Panel_Collision: Flag for panel collision
            - Panel_Xray_Collision: Flag for panel-xray collision
            - Outside_Wafer: Flag for geometry outside wafer
            - Acceptable_Stray: Flag for acceptable stray radiation levels
            - Contained_Panels: Panel containment distance
        """
        
        def _get_mask(x_grid, y_grid, width):
            """Create boolean masks for a square region centered at origin."""
            x_mask = (x_grid > -width/2) & (x_grid < width/2)
            y_mask = (y_grid > -width/2) & (y_grid < width/2)
            return x_mask, y_mask
        
        def _calculate_overlap_2d(ir_map, x_grid, y_grid, width, n_emitters):
            """Calculate 2D overlap percentage within specified width."""
            x_mask, y_mask = _get_mask(x_grid, y_grid, width)
            inside_area = x_mask & y_mask
            overlap = 100.0 * np.sum(ir_map[inside_area]) / np.sum(inside_area) / n_emitters
            return overlap
        
        def _calculate_metrics(ir_map_e, ir_map_p, ir_map):
            """Calculate all performance metrics for the detector."""
            # Set up coordinate grids
            x_range = self.x_range
            y_range = self.y_range
            length_x = len(x_range)
            length_y = len(y_range)
            
            x_grid = np.repeat(x_range, length_y).reshape(length_x, length_y).T
            y_grid = np.repeat(np.flip(y_range), length_x).reshape(length_y, length_x)
            
            # Create detector masks
            x_mask, y_mask = _get_mask(x_grid, y_grid, d_width)
            inside_detector = x_mask & y_mask
            outside_detector = ~inside_detector
            inside_detector_size = int(np.sqrt(np.sum(inside_detector)))
            
            # Extract detector pixels
            detector_pixels = ir_map[inside_detector]
            stray_pixels = ir_map[outside_detector]
            
            # Calculate stray radiation
            total_stray = np.sum(stray_pixels) / (self.pix_per_cm ** 2)
            
            # Calculate detector coverage (illumination)
            n_detector_pixels = len(np.ravel(detector_pixels))
            detector_coverage = 100.0 * np.sum((detector_pixels > 0).astype(int)) / n_detector_pixels
            
            # Calculate 2D overlap at detector surface
            overlap_2d = _calculate_overlap_2d(ir_map, x_grid, y_grid, d_width, len(ir_map_e))
            
            # Calculate 2D overlap at mid-layer
            if len(self.intersection_heights) == 2:
                mid_layer_index = 1
            else:
                mid_layer_index = 14
            
            mid_layer_ir_map = self.irradiation_layers[mid_layer_index]
            mid_overlap_2d = _calculate_overlap_2d(mid_layer_ir_map, x_grid, y_grid, d_width, len(ir_map_e))
            
            # Calculate 3D volumetric overlap
            inside_detector_volume = np.tile(inside_detector, (self.irradiation_layers.shape[0] - 1, 1, 1))
            pvolume = self.irradiation_layers[1:, :, :]
            overlap_3d = (100.0 * np.sum(pvolume[inside_detector_volume]) / 
                        np.sum(inside_detector_volume) / len(ir_map_e))
            
            # Calculate angular range between emitters
            angular_range = _calculate_angular_range(ir_map_e, inside_detector, inside_detector_size)
            
            return detector_coverage, overlap_2d, mid_overlap_2d, angular_range, total_stray, overlap_3d
        
        def _calculate_angular_range(ir_map_e, inside_detector, inside_detector_size):
            """Calculate maximum angular separation between emitter pairs."""
            n_emitters = ir_map_e.shape[0]
            
            if n_emitters <= 1:
                return 0.0
            
            # Apply detector mask to all emitters
            inside_detector_all = np.tile(inside_detector, (n_emitters, 1, 1))
            ir_map_e_detector = ir_map_e[inside_detector_all]
            ir_map_e_detector = ir_map_e_detector.reshape(n_emitters, inside_detector_size, inside_detector_size)
            
            # Get emitter positions
            all_emitter_positions = np.concatenate(self.cone_positions, axis=0)
            
            # Find minimum cosine (maximum angle) between emitter pairs
            min_cosine = 1e15
            pair_counter = 0
            
            for i in range(n_emitters):
                for j in range(i + 1, n_emitters):
                    # Check for overlapping coverage
                    area_and = np.sum(np.logical_and(ir_map_e_detector[i], ir_map_e_detector[j]))
                    
                    if area_and != 0:
                        # Calculate angle between emitters relative to detector center
                        cosine = (np.dot(all_emitter_positions[i], all_emitter_positions[j]) /
                                (np.linalg.norm(all_emitter_positions[i]) * 
                                np.linalg.norm(all_emitter_positions[j])))
                        
                        if cosine < min_cosine:
                            min_cosine = cosine
                    
                    pair_counter += 1
            
            # Verify all pairs were counted
            expected_pairs = (n_emitters - 1) * n_emitters // 2
            if pair_counter != expected_pairs:
                raise Exception("Emitter pairs not counted correctly.")
            
            # Convert to degrees
            angular_range = np.arccos(min_cosine) / np.pi * 180.0
            return angular_range
        
        # Check for invalid geometry

        phys_unrealism = (self.panel_collision + self.panelXRay_collision + self.outside_wafer) >= 1
        
        if phys_unrealism:
            if not self.silence_tqdm:
                print('Invalid geometry, data set to NaNs')
            return
        
        if not self.acceptable_stray_at_detector:
            if not self.silence_tqdm:
                print('Too much stray radiation, outputs set to NaNs')
            return
        
        # Validate detector alignment with pixel grid
        self.d_width = d_width
        half_detector_pix = self.pix_per_cm * 1e2 * d_width / 2
        
        if int(half_detector_pix) != half_detector_pix:
            if not self.silence_tqdm:
                print('WARNING! Detector boundary does not align with pixel boundary. '
                    'Recommend reconsidering pixel resolution.')
        
        # Get irradiation maps
        ir_map = self.irradiation_layers[0]
        ir_map_e = self.irradiation_layers_e[0]
        ir_map_p = self.irradiation_layers_p[0]
        
        # Calculate metrics
        (illumination, overlap_2d, mid_overlap_2d, 
        angular_range, total_stray, overlap_3d) = _calculate_metrics(ir_map_e, ir_map_p, ir_map)
        
        # Compile results into DataFrame
        simulation_data = np.vstack((
            self.cone_angle, self.SID, self.emitter_pitch, self.panel_pitch, self.panel_theta,
            angular_range, illumination, overlap_2d, mid_overlap_2d, overlap_3d, total_stray,
            self.panel_collision, self.panelXRay_collision, int(self.outside_wafer), 
            self.acceptable_stray_at_detector, self.contained_panel_dist
        )).T
        
        column_names = [
            'Cone_Angle', 'SID', 'Emitter_Pitch', 'Panel_Pitch', 'Panel_Angle',
            'Angular_Range', 'Illumination', 'Overlap_2D', 'Overlap_Mid_2D', 'Overlap_3D',
            'Stray_Radiation', 'Panel_Collision', 'Panel_Xray_Collision', 'Outside_Wafer',
            'Acceptable_Stray', 'Contained_Panels'
        ]
        
        self.df = pd.DataFrame(simulation_data, columns=column_names)

    def PlotMaps(self,
                d_width = 0.43, 
                plot_level = None,
                plot_lims = [-0.3, 0.3],
                save_plot = True,
                plot_title = None,
                plot_filename = None):

        """
        Visualise 2D irradiation maps across one or more detector planes.

        Parameters
        ----------
        d_width : float, optional
            Width (in metres) of the square detector area centered at (0, 0).
            Default is 0.43 m.
        plot_level : str or None, optional
            Determines which maps to plot:
            - 'all' : plots all intersection heights
            - 'detector only' : plots only the z=0 (detector) plane
            - None : no plotting (default)
        plot_lims : list of float, optional
            [min, max] limits (in metres) for both x and y axes. Default is [-0.3, 0.3].
        save_plot : bool, optional
            If True, saves plots as PNGs; otherwise displays them interactively. Default is True.
        plot_title : str or None, optional
            Custom title for plots. If None, defaults to plane height information.
        plot_filename : str or None, optional
            If provided, saves all plots to this path/prefix. Otherwise, filenames are autogenerated.

        Notes
        -----
        - Uses `self.irradiation_layers`, `self.x_range_edges`, and `self.vmax` from prior computation.
        - Colorbar indicates number of active emitters per pixel ($N_{em}/pix$).
        """

        if not hasattr(self, "irradiation_layers"):
            raise AttributeError("No irradiation data found. Run Irradiation() first.")
            
        def plot_illumination(k):

            rainbow = cm.get_cmap('rainbow',226)
            newcolors = rainbow(np.linspace(0,1,226))
            newcolors[0,:] = np.array([1.,1.,1.,1])
            newcmp = colors.ListedColormap(newcolors)

            ir_map = self.irradiation_layers[k]
            detector = patches.Rectangle((-d_width/2,-d_width/2), d_width, d_width, linewidth=1, edgecolor='k', facecolor='none')
            limit = patches.Rectangle((-(d_width+(0.03*self.SID))/2,-(d_width+(0.03*self.SID))/2), d_width+(0.03*self.SID), d_width+(0.03*self.SID), linewidth=1, edgecolor='limegreen', facecolor='none')
            fig, axs = plt.subplots(1,1, figsize = [6,6])
            im = axs.imshow(ir_map, origin = 'lower', extent = [self.x_range_edges[0], self.x_range_edges[-1], self.y_range_edges[0], self.y_range_edges[-1]], cmap = newcmp, vmax =self.vmax )#tol_cmap('sunset'), vmax =self.vmax )
            for i in range(0, len(self.x_range_edges)):
                xed = self.x_range_edges[i]
                axs.plot([xed, xed],[-1,1], linewidth = 0.2, color = 'darkslategrey', alpha = 0.3)
                axs.plot([-1,1], [xed, xed], linewidth = 0.2, color = 'darkslategrey', alpha = 0.3)
            axs.set_xlim(plot_lims)
            axs.set_ylim(plot_lims)
            axs.add_patch(detector)

            axs.xaxis.set_tick_params(direction= 'in', which='both', right =True, top = True)
            axs.yaxis.set_tick_params(direction= 'in', which='both', right =True, top = True)

            xticks = np.linspace(plot_lims[0], plot_lims[1], 7)
            axs.set_xticks(xticks)
            axs.set_xticklabels([f"{x*100:.1f}" for x in xticks])

            yticks = np.linspace(plot_lims[0], plot_lims[1], 7)
            axs.set_yticks(yticks)
            axs.set_yticklabels([f"{y*100:.1f}" for y in yticks])

            axs.set_xlabel('$x$-extent [cm]')
            axs.set_ylabel('$y$-extent [cm]')
            X2D,Y2D = np.meshgrid(self.x_range, self.y_range)
            out = np.column_stack((X2D.ravel(),Y2D.ravel())).T
            xs, ys = out[0], out[1]

            d_2 = d_width / 2

            axs.scatter([-d_2, -d_2, d_2, d_2], [-d_2, d_2, -d_2, d_2], color = 'blue', marker = 'x', s = 10)

            top_pos = axs.get_position()
            bottom_pos = axs.get_position()

            colorbar_left = 0.86
            colorbar_bottom = bottom_pos.y0
            colorbar_width = 0.04
            colorbar_height = top_pos.y1 - bottom_pos.y0
            colorbar_rect = [colorbar_left, colorbar_bottom, colorbar_width, colorbar_height]

            n_emitters = self.grid**2 * sum(self.activate)
            n_ticks = 5
            ticks = np.linspace(1, n_emitters, n_ticks, dtype=int)
            
            cbax = fig.add_axes(colorbar_rect)
            cbar = plt.colorbar(im, cax = cbax, label = '$N_\mathrm{em}/\mathrm{pix}$', ticks=ticks)
            
            Line2D = lines.Line2D
            custom_lines = [Line2D([0], [0], color='white', linestyle='-',alpha=0.1),
                            Line2D([0], [0], color='white', linestyle='-',alpha=0.1)]

            sid = round(self.SID, 2)
            height = round(self.intersection_heights[k], 2)
            idx = int(round((sid - 0.3)/0.01, 1))
            if plot_title is None:
                axs.set_title('Ir Height = %s m' % (height))
            else:
                axs.set_title(plot_title)
            if save_plot:
                if plot_filename is None:
                    plt.savefig('illumination_at_z_%s_idx_%s.png' % (self.intersection_heights[k], idx))
                else:
                    plt.savefig(plot_filename)
            else:
                plt.show()
            plt.close()

        if plot_level == 'detector only':
            plot_illumination(0)
        else:
            for k in range(0, len(self.intersection_heights)):
                plot_illumination(k)
        
    def Visualise(self, 
                  include_patient = True):
        """
        Visualise the current X-ray scanner setup using VTK.

        This method renders all VTK actors associated with the scanner geometry —
        including panels, emitters, X-ray cones, and optionally a human body model.
        It supports a live VTK interactor.

        Parameters
        ----------
        include_patient : bool, optional
            If True, includes a scaled anatomical body model in the scene for context.
            Default is True.
        Notes
        -----
        - If the geometry is invalid and `force_vis` is False, no visualisation
        will be generated.
        """

        # --- Determine geometry validity ---
        reasons = []
        if self.panel_collision:
            reasons.append("panel collision")
        if self.panelXRay_collision:
            reasons.append("panel irradiated by X-rays")
        if self.outside_wafer:
            reasons.append("emitters outside wafer bounds")

        # Classify geometry
        if not reasons:
            geom_status = "Acceptable geometry"
        elif not self.force_vis:
            geom_status = "Invalid geometry"
        else:
            geom_status = "Invalid geometry (visualisation forced)"

        reason_str = f" ({', '.join(reasons)})" if reasons else ""

        # --- Exit early if invalid and not forced ---
        if reasons and not self.force_vis:
            print(f"{geom_status}{reason_str}, no visualisation created.")
            return

        print(f"{geom_status}{reason_str}, creating visualisation...")

        # --- Setup renderer and add base detector ---
        renderer = vtk.vtkRenderer()
        renderer.AddActor(Cuboid())  # Detector representation

        # Add additional actors (e.g., environment, markers)
        for actor in self.additional_actors:
            renderer.AddActor(actor)

        # --- Add all panels and associated emitters/probes ---
        for panel, emitters, plane_probes, patient_probes in zip(
            self.panel_actors,
            self.emitter_actors,
            self.plane_probe_master_list,
            self.patient_probe_master_list,
        ):
            renderer.AddActor(panel)
            for emitter_actor, plane_probe in zip(emitters, plane_probes):
                renderer.AddActor(plane_probe)
                renderer.AddActor(emitter_actor)

            for patient_probe in patient_probes:
                renderer.AddActor(patient_probe)

        # --- Optional patient model ---
        if include_patient:
            body_actors, _ = read_body(scale=0.09, rotate=(270, 0, 0), translate=(1.8, -0.25, -1.15))
            for item in body_actors.values():
                if isinstance(item, dict):
                    for sub_actor in item.values():
                        renderer.AddActor(sub_actor)
                else:
                    renderer.AddActor(item)

        # --- Render window and visual settings ---
        window = vtk.vtkRenderWindow()
        window.AddRenderer(renderer)
        window.SetSize(1000, 1000)
        window.PolygonSmoothingOn()
        window.LineSmoothingOn()
        window.PointSmoothingOn()

        # Background and lighting
        renderer.GradientBackgroundOn()
        renderer.SetBackground(0.9, 0.9, 0.9)
        renderer.SetBackground2(0.9, 0.9, 0.9)
        renderer.AddActor(Axes(renderer))

        # Camera and rendering passes
        camera = renderer.GetActiveCamera()
        camera.SetClippingRange(0, 1)
        renderer.UseDepthPeelingOn()
        renderer = interactor_utils.set_passes(renderer, use_ssao=True)
        renderer.ResetCamera()

        # --- Interaction setup ---
        window.Render()
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(window)
        interactor.SetInteractorStyle(
            interactor_utils.MyInteractorStyle(
                interactor, camera, window
            )
        )
        interactor_utils.add_camera_widget(renderer)
        interactor.Start()

