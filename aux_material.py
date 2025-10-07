import vtk
import numpy as np
from read_patient import *
from scipy.spatial.transform import Rotation as R
from vtk.util import numpy_support
from tqdm import tqdm
from matplotlib.cm import get_cmap
import clipping_utils

DC_arrangement = np.array( [1, 0, 1,
                            0, 1, 0,
                            1, 0, 1])

D_arrangement =  np.array( [1, 0, 1,
                            0, 0, 0,
                            1, 0, 1])

P_arrangement  = np.array( [0, 1, 0, 
                            1, 0, 1, 
                            0, 1, 0])

PC_arrangement   = np.array([0, 1, 0, 
                             1, 1, 1, 
                             0, 1, 0])

all_panels   = np.ones(9)

full_intersection_heights = np.concatenate(([0.0], np.linspace(0.0, 0.27, 28) + np.diff(np.arange(0, 0.25, 0.01))[0]/2))
detector_height = [0.0]
detector_and_mid = np.array((0.0, 0.135))

def Axes(renderer, name='axes'):

    axes = vtk.vtkCubeAxesActor()
    axes.SetCamera(renderer.GetActiveCamera())
    axes.GetProperty().SetColor((0,0,0))
    # axes.SetFlyModeToFurthestTriad()
    axes.SetFlyModeToOuterEdges()
    # axes.SetFlyModeToStaticEdges()
    axes_units = 'm'
    axes.SetXTitle('X-Axis')
    axes.SetXUnits(axes_units)
    axes.GetXAxesLinesProperty().SetColor(0.0, 0.0, 0.0)
    axes.GetTitleTextProperty(0).SetColor(0.0, 0.0, 0.0)
    axes.GetLabelTextProperty(0).SetColor(0.0, 0.0, 0.0)

    axes.SetYTitle('Y-Axis')
    axes.SetYUnits(axes_units)
    axes.GetYAxesLinesProperty().SetColor(0.0, 0.0, 0.0)
    axes.GetTitleTextProperty(1).SetColor(0.0, 0.0, 0.0)
    axes.GetLabelTextProperty(1).SetColor(0.0, 0.0, 0.0)

    axes.SetZTitle('Z-Axis')
    axes.SetZUnits(axes_units)
    axes.GetZAxesLinesProperty().SetColor(0.0, 0.0, 0.0)
    axes.GetTitleTextProperty(2).SetColor(0.0, 0.0, 0.0)
    axes.GetLabelTextProperty(2).SetColor(0.0, 0.0, 0.0)

    axes.GetXAxesGridlinesProperty().SetColor(0.0, 0.0, 0.0)
    axes.GetYAxesGridlinesProperty().SetColor(0.0, 0.0, 0.0)
    axes.GetZAxesGridlinesProperty().SetColor(0.0, 0.0, 0.0)

    axes.DrawXGridlinesOn()
    axes.DrawYGridlinesOn()
    axes.DrawZGridlinesOn()
    axes.SetGridLineLocation(axes.VTK_GRID_LINES_FURTHEST)

    axes.SetObjectName(name)
    axes.Modified()

    return axes

def Cuboid(centre=[0,0,0], z_height=0.01, side_length=2, name='detector', opacity=0.2):
    # Create a cuboid
    cuboid = vtk.vtkCubeSource()
    offset = np.asarray([0, 0, z_height/2])
    new_centre = np.asarray(centre) + offset
    cuboid.SetCenter(new_centre)
    cuboid.SetXLength(side_length)
    cuboid.SetYLength(side_length)
    cuboid.SetZLength(z_height)
    cuboid.Update()

    # Create a mapper and actor for the cuboid
    cuboidMapper = vtk.vtkPolyDataMapper()
    cuboidMapper.SetInputConnection(cuboid.GetOutputPort())

    cuboidActor = vtk.vtkActor()
    cuboidActor.SetMapper(cuboidMapper)
    cuboidActor.GetProperty().SetOpacity(opacity)
    cuboidActor.SetObjectName(name)
    return cuboidActor

def Plane(height):
    plane = vtk.vtkPlane()
    plane.SetOrigin(0.0, 0.0, -1/2)
    plane.SetNormal(0, 0, 1)
    plane.Push(height)
    return plane

def nan_array(shape):
    nan_array = np.zeros(shape)
    nan_array[:] = np.nan
    return nan_array

def transform_data(mat, data):
    #create transformation matrix
    rot_mat = vtk.vtkMatrix4x4()
    for i in range(4):
                  for j in range(4):
                      rot_mat.SetElement(i, j, mat[i, j])
                      
    # Create a vtkTransform to apply the transformation matrix
    transform = vtk.vtkTransform()
    transform.SetMatrix(rot_mat)
    
    # Apply the transformation to the polydata
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(data)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    
    # Get the rotated polydata
    transformed_polydata = transform_filter.GetOutput()
    return transformed_polydata

def get_transformation_matrices(angle, rotation_axis, point):
    """
    mat T1 - translation matrix to move (0,0,0) to point
    matR - rotation matrix to rotate by angle around rotation_axis,
    matT2 -  translation matrix to  move backto orginal center
    """
    alpha = rotation_axis[0]*angle
    beta = rotation_axis[1]*angle
    gamma = rotation_axis[2]*angle
    matR = np.matrix(
                  [
                      [np.cos(beta)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma)-np.cos(alpha)*np.sin(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma)+np.sin(alpha)*np.sin(gamma), 0],
                      [np.cos(beta)*np.sin(gamma), np.sin(alpha)*np.sin(beta)*np.sin(gamma)+np.cos(alpha)*np.cos(gamma), np.cos(alpha)*np.sin(beta)*np.sin(gamma)-np.sin(alpha)*np.cos(gamma), 0],
                      [-np.sin(beta), np.sin(alpha)*np.cos(beta), np.cos(alpha)*np.cos(beta), 0],
                      [0,0,0,1]
                      ]
                  )
        
    matT1 = np.matrix([[1, 0, 0, -point[0]], [ 0, 1 , 0 , -point[1]], [0, 0, 1 ,-point[2]],[ 0, 0 ,0 ,1 ]])
    matT2 = np.matrix([[1, 0, 0, point[0]], [ 0, 1 , 0 , point[1]], [0, 0, 1 , point[2]],[ 0, 0 ,0 ,1 ]])
    return matT1, matR, matT2    

class Emitter():

    """
    Represents a conical X-ray emitter defined by its geometry.

    Parameters
    ----------
    sides : int, optional
        Number of sides used to approximate the conical surface 
        (default is 4, i.e. a square cone).
    cone_angle : float, optional
        Full cone opening angle in degrees (default is 12.0).
    height : float, optional
        Height of the emitter cone (default is 1.0, needs to be large enough to 
                                    intersect with the detector plane).

    Attributes
    ----------
    resolution : int
        Number of sides approximating the cone.
    cone_angle : float
        Full opening angle of the cone in degrees.
    height : float
        Height of the emitter cone.
    radius : float
        Base radius of the cone computed from the height and cone angle.
    """

    def __init__(self, sides = 4, cone_angle = 12., height = 1.):
        self.resolution = sides
        self.cone_angle = cone_angle 
        opening_angle_radians = np.deg2rad(cone_angle)
        self.height = height
        self.radius = height * np.tan(opening_angle_radians / 2.0)

class SinglePanel:
    """
    Represents a single flat panel with an array of X-ray emitters.
    
    The panel can be positioned at various locations (center, edges, corners) and
    rotated to specific angles. Emitters are arranged in a square grid pattern.
    
    Parameters
    ----------
    SID : float, optional
        Source-to-image distance in meters. Default is 1.0.
    centre : tuple of float, optional
        Panel center coordinates (x, y, z). If None, defaults to (0, 0, SID).
    emitter_pitch : float, optional
        Spacing between emitters in meters. Default is 0.025.
    grid : int, optional
        Grid size (grid x grid emitters). Default is 4.
    cone_angle : float, optional
        X-ray cone emission angle in degrees. Default is 12.
    cone_resolution : int, optional
        Number of sides for cone mesh representation. Default is 4.
    emission_length : float, optional
        Height/length of emission cone in meters. Default is 1.
    panel_name : str, optional
        Panel position identifier. Options: 'centre', 'top', 'bottom', 'left', 'right',
        'top_left', 'top_right', 'bottom_left', 'bottom_right'. Default is 'centre'.
    panel_theta : float, optional
        Panel rotation angle in degrees. Default is 6.
    panel_alpha : float, optional
        Panel rotation around z-axis in degrees. Default is 0.
    cone_alpha : float, optional
        Cone rotation around z-axis in degrees. Default is 0.
    silence_tqdm : bool, optional
        If True, suppress progress bar output. Default is True.
    
    Attributes
    ----------
    n_emitters : int
        Total number of emitters (grid²).
    rot_axis : ndarray
        Rotation axis for panel positioning.
    cuboid_object : vtk.vtkCubeSource or vtk.vtkCylinderSource
        VTK object representing panel body.
    cuboid_actor : vtk.vtkActor
        VTK actor for rendering panel body.
    panel_normal : ndarray
        Normal vector of panel surface.
    cone_actors : list of vtk.vtkActor
        VTK actors for rendering emitter cones.
    cone_positions : list of ndarray
        World coordinates of emitter positions.
    centred_emitters : ndarray
        Local coordinates of emitters before transformation.
    """
    
    # Panel position configurations (rotation axes)
    PANEL_ROTATION_AXES = {
        'top_left':     np.array([-1., -1., 0.]),
        'top':          np.array([-1.,  0., 0.]),
        'top_right':    np.array([-1.,  1., 0.]),
        'left':         np.array([ 0., -1., 0.]),
        'centre':       np.array([ 0.,  0., 0.]),
        'right':        np.array([ 0.,  1., 0.]),
        'bottom_left':  np.array([ 1., -1., 0.]),
        'bottom':       np.array([ 1.,  0., 0.]),
        'bottom_right': np.array([ 1.,  1., 0.])
    }
    
    def __init__(self, 
                 SID=1.0, 
                 centre=None, 
                 emitter_pitch=0.025, 
                 grid=4,
                 cone_angle=12.0,
                 cone_resolution=4,
                 emission_length=1.0, 
                 panel_name='centre',
                 panel_theta=6.0,
                 panel_alpha=0.0,
                 cone_alpha=0.0,
                 silence_tqdm=True):
        
        # Panel parameters
        self.SID = SID
        self.centre = centre if centre is not None else (0.0, 0.0, SID)
        self.panel_name = panel_name
        self.panel_theta = panel_theta  # Panel tilt angle with respect to rotation axis
        self.panel_alpha = panel_alpha  # Panel rotation about z-axis
        
        # Emitter/x-ray cone parameters
        self.emission_length = emission_length
        self.emitter_pitch = emitter_pitch
        self.grid = grid
        self.n_emitters = grid ** 2
        self.cone_angle = cone_angle
        self.cone_alpha = cone_alpha
        self.cone_resolution = cone_resolution
        
        # User settings
        self.silence_tqdm = silence_tqdm
        
        # Initialise storage for VTK objects
        self._initialise_vtk_storage()
        
    def _initialise_vtk_storage(self):
        """Initialise empty containers for VTK objects."""
        self.rot_axis = np.array([])
        self.panel_normal = np.array([])
        
        self.cuboid_object = None
        self.cuboid_actor = None
        
        self.cone_actors = []
        self.cone_objects = []
        self.bounding_objects = []
        self.bounding_actors = []
        self.cone_positions = []
        self.centred_emitters = None
        
    def _get_rotation_axis(self):
        """
        Get the rotation axis for the panel based on its position name.
        
        Returns
        -------
        ndarray
            Normalised rotation axis vector.
        """
        rot_axis = self.PANEL_ROTATION_AXES[self.panel_name].copy()
        
        # Normalise if not the center panel
        if self.panel_name != "centre":
            rot_axis = rot_axis / np.linalg.norm(rot_axis)
        
        return rot_axis
    
    def _get_panel_normal(self, world_centre=(0.0, 0.0, 0.0)):
        """
        Calculate the normal vector of the panel surface.
        
        Parameters
        ----------
        world_centre : tuple of float, optional
            World origin coordinates. Default is (0, 0, 0).
        
        Returns
        -------
        ndarray
            Panel normal vector pointing from world center to panel center.
        """
        return np.array(self.centre) - np.array(world_centre)
    
    def _calculate_rotation_point_for_imported_panel(self):
        """
        Calculate the point around which to rotate an imported CAD panel.
        
        This method is specific to imported panel geometries and determines
        the rotation point based on panel position.
        
        Returns
        -------
        tuple of float
            (x, y, z) coordinates of rotation point.
        """
        # Hard-coded bounds for panel object, based on a model design
        bounds = np.array([
            -0.10499998927116394, 0.10499998927116394,  # y bounds
            -0.10499998927116394, 0.10499998927116394,  # x bounds
            -0.023000001907348633, 0.023000001907348633  # z bounds
        ])
        

        k = np.cos(np.deg2rad(45))  # Factor for diagonal positions
        
        # Map panel positions to rotation points
        rotation_points = {
            'top':          (0, bounds[0], bounds[5]),
            'left':         (bounds[3], 0, bounds[5]),
            'right':        (bounds[2], 0, bounds[5]),
            'bottom':       (0, bounds[1], bounds[5]),
            'top_left':     (k * bounds[3], k * bounds[0], bounds[5]),
            'top_right':    (k * bounds[2], k * bounds[0], bounds[5]),
            'bottom_left':  (k * bounds[3], k * bounds[1], bounds[5]),
            'bottom_right': (k * bounds[2], k * bounds[1], bounds[5]),
            'centre':       (0, 0, 0)
        }
        
        return rotation_points.get(self.panel_name, (0, 0, 0))
    
    def ConstructPanelBody(self, name='panel'):
        """
        Construct and position a cylindrical panel body from design specifications.
        
        Creates a cylindrical panel body with hard-coded dimensions, then 
        rotates and positions it according to panel configuration.
        
        Parameters
        ----------
        name : str, optional
            Object name for the panel actor. Default is 'panel'.
        """
        # Hard-coded panel dimensions 
        panel_thickness = 0.05
        panel_width = 0.21
        wafer_size = 0.1
        
        # Create cylindrical panel
        cylinder = vtk.vtkCylinderSource()
        cylinder.SetRadius(panel_width / 2)
        cylinder.SetHeight(panel_thickness)
        cylinder.SetResolution(20)
        cylinder.Update()
        
        center = cylinder.GetCenter()
        
        # Rotate cylinder 90° around x-axis to align with coordinate system
        rotation_axis = (1.0, 0.0, 0.0)
        matT1, matR, matT2 = get_transformation_matrices(
            np.deg2rad(90.0), rotation_axis, center
        )
        mat = matR * matT1
        rotated_cylinder = transform_data(mat, cylinder.GetOutput())
        
        # Apply panel-specific rotation and positioning
        point = self._calculate_rotation_point_for_imported_panel()
        self.rot_axis = self._get_rotation_axis()
        
        matT1, matR, matT2 = get_transformation_matrices(
            np.deg2rad(self.panel_theta), self.rot_axis, point
        )
        
        # Translation matrix to final position
        matT3 = np.matrix([
            [1, 0, 0, self.centre[0]],
            [0, 1, 0, self.centre[1]],
            [0, 0, 1, self.centre[2] + panel_thickness / 2],
            [0, 0, 0, 1]
        ])
        
        mat = matT3 * matT2 * matR * matT1
        rotated_polydata_cylinder = transform_data(mat, rotated_cylinder)
        
        # Create VTK actor
        panelMapper = vtk.vtkPolyDataMapper()
        panelMapper.SetInputData(rotated_polydata_cylinder)
        
        panel_actor = vtk.vtkActor()
        panel_actor.SetMapper(panelMapper)
        panel_actor.SetObjectName(name)
        panel_actor.GetProperty().SetOpacity(0.5)
        panel_actor.GetProperty().SetColor((0.9, 0.9, 0.9))
        
        # Store panel properties
        self.panel_thickness = panel_thickness
        self.point_to_rotate = point
        self.mat = mat
        self.cuboid_actor = panel_actor
        self.wafer_size = wafer_size
    
    
    def _rotate_points(self, points, axis, theta):
        """
        Rotate points around an axis using Rodrigues' rotation formula.
        
        Parameters
        ----------
        points : ndarray
            Array of 3D points to rotate, shape (N, 3).
        axis : ndarray
            Rotation axis vector.
        theta : float
            Rotation angle in radians.
        
        Returns
        -------
        rotated_points : ndarray
            Rotated points, shape (N, 3).
        normals : ndarray
            Normal vectors after rotation, shape (N, 3).
        y_angles : ndarray
            Angles relative to y-plane in degrees.
        x_angles : ndarray
            Angles relative to x-plane in degrees.
        """
        normal = np.array([0, 0, -1])
        x_plane = np.array([1, 0, 0])
        y_plane = np.array([0, 1, 0])
        
        # Special case: zero rotation axis
        if np.allclose(axis, [0, 0, 0]):
            x_angles = np.rad2deg(self._angle_between(
                normal, np.tile(x_plane, (len(points), 1))
            ))
            y_angles = np.rad2deg(self._angle_between(
                normal, np.tile(y_plane, (len(points), 1))
            ))
            return points, normal, y_angles, x_angles
        
        # Normalise rotation axis
        k = axis / np.linalg.norm(axis)
        
        costh = np.cos(theta)
        sinth = np.sin(theta)
        
        # Compute rotation matrix
        k_cross = np.array([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
        ])
        k_outer = np.outer(k, k)
        R = costh * np.eye(3) + sinth * k_cross + (1 - costh) * k_outer
        
        # Apply rotation
        rotated_points = np.dot(R, points.T).T
        rotated_normals = np.dot(normal, R.T)
        rotated_normals = np.tile(rotated_normals, (len(points), 1))
        
        # Calculate angles relative to coordinate planes
        x_angles = np.rad2deg(self._angle_between(
            rotated_normals, np.tile(x_plane, (len(points), 1))
        ))
        y_angles = np.rad2deg(self._angle_between(
            rotated_normals, np.tile(y_plane, (len(points), 1))
        ))
        
        return rotated_points, rotated_normals, y_angles, x_angles
    
    @staticmethod
    def _angle_between(v1, v2):
        """
        Compute angle between n-dimensional vectors.
        
        Parameters
        ----------
        v1 : ndarray
            First vector or array of vectors.
        v2 : ndarray
            Second vector or array of vectors.
        
        Returns
        -------
        ndarray
            Angle(s) in radians.
        """
        cos_theta = (np.sum(v1 * v2, axis=-1) / 
                     (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)))
        return np.arccos(cos_theta)
    
    def _generate_grid_positions(self):
        """
        Generate emitter positions in a square grid pattern.
        
        Returns
        -------
        ndarray
            Array of emitter positions in local coordinates, shape (n_emitters, 3).
        """
        n = int(np.sqrt(self.n_emitters))
        
        # Create grid indices
        indices = np.arange(1, 2 * n + 1, 2)
        offset = n
        
        positions = []
        for col_idx in indices:
            for row_idx in indices:
                x = (col_idx - offset) / 2 * self.emitter_pitch
                y = (row_idx - offset) / 2 * self.emitter_pitch
                positions.append([x, y, 0])
        
        return np.array(positions)
    
    def EmitterArray(self, emitter_colour=(0.0, 0.5, 0.5), 
                     panel_number=0):
        """
        Create and position array of X-ray emitter cones.
        
        Generates cone geometries representing X-ray emitters arranged in a grid,
        applies panel rotations, and creates VTK actors for visualization.
        
        Parameters
        ----------
        emitter_colour : tuple of float, optional
            RGB color for emitter visualization (0-1 range). Default is (0.0, 0.5, 0.5).
        panel_number : int, optional
            Panel identifier for naming. Default is 0.
        """
        # Generate base emitter positions
        locs_centred = self._generate_grid_positions()
        
        # Apply panel alpha rotation (around z-axis)
        if self.panel_alpha != 0:
            locs_centred = self._apply_z_rotation(locs_centred, self.panel_alpha)
        
        # Create emitter geometry
        E = Emitter(
            sides=self.cone_resolution,
            cone_angle=self.cone_angle,
            height=self.emission_length
        )
        
        # Create base cone source
        coneSource = vtk.vtkConeSource()
        coneSource.SetCenter(0, 0, 0)
        coneSource.SetRadius(E.radius)
        coneSource.SetHeight(E.height)
        coneSource.SetDirection([0, 0, 1])
        coneSource.SetResolution(E.resolution)
        coneSource.Update()
        
        # Generate cone actors for each emitter position
        cone_actors = []
        cone_positions_world = []
        
        for i, local_pos in enumerate(locs_centred):
            cone_actor, world_pos = self._create_emitter_cone(
                coneSource, local_pos, E.height, emitter_colour
            )
            cone_actor.SetObjectName(f'panel-{panel_number}/cones/cone-{i}')
            cone_actors.append(cone_actor)
            cone_positions_world.append(world_pos)
        
        # Store results
        self.cone_actors = cone_actors
        self.cone_positions = cone_positions_world
        self.centred_emitters = locs_centred
        self.cone_objects = []
        self.bounding_objects = []
        self.bounding_actors = []
    
    def _apply_z_rotation(self, points, angle_degrees):
        """
        Apply rotation around z-axis to points.
        
        Parameters
        ----------
        points : ndarray
            Array of 3D points, shape (N, 3).
        angle_degrees : float
            Rotation angle in degrees.
        
        Returns
        -------
        ndarray
            Rotated points, shape (N, 3).
        """
        matT1, matR, matT2 = get_transformation_matrices(
            np.deg2rad(angle_degrees),
            (0.0, 0.0, 1.0),
            [0, 0, 0]
        )
        mat = matT2 * matR * matT1
        
        # Convert to homogeneous coordinates
        homogeneous = np.column_stack((points, np.ones(len(points))))
        rotated = np.dot(homogeneous, mat.T)
        
        return rotated[:, :3]
    
    def _create_emitter_cone(self, cone_source, local_position, height, colour):
        """
        Create a single emitter cone actor at specified position.
        
        Parameters
        ----------
        cone_source : vtk.vtkConeSource
            Base cone geometry.
        local_position : ndarray
            Position in local panel coordinates.
        height : float
            Cone height in meters.
        colour : tuple of float
            RGB color values.
        
        Returns
        -------
        cone_actor : vtk.vtkActor
            VTK actor representing the cone.
        world_position : list
            Cone position in world coordinates.
        """
        # Translate cone tip to emitter position
        point = [0, 0, height / 2 + self.panel_thickness / 2] - local_position
        matT1 = np.matrix([
            [1, 0, 0, -point[0]],
            [0, 1, 0, -point[1]],
            [0, 0, 1, -point[2]],
            [0, 0, 0, 1]
        ])
        
        # Calculate world position of cone
        cone_pos_homogeneous = np.array([0, 0, height / 2, 1.0])
        cone_pos_translated = np.dot(matT1, cone_pos_homogeneous)
        cone_pos_world = np.dot(self.mat, np.array(cone_pos_translated)[0])
        world_position = np.array(cone_pos_world)[0][:3].tolist()
        
        # Apply transformations to geometry
        polydata = cone_source.GetOutput()
        translated_data = transform_data(matT1, polydata)
        
        # Rotate cone around z-axis
        rot_axis = (0.0, 0.0, 1.0)
        angle = np.deg2rad(self.cone_alpha + self.panel_alpha)
        matT1, matR, matT2 = get_transformation_matrices(
            angle, rot_axis, local_position
        )
        mat = matT2 * matR * matT1
        translated_rotated_data = transform_data(mat, translated_data)
        
        # Apply panel transformation
        rotated_polydata = transform_data(self.mat, translated_rotated_data)
        
        # Create VTK actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(rotated_polydata)
        
        cone_actor = vtk.vtkActor()
        cone_actor.SetMapper(mapper)
        cone_actor.GetProperty().SetOpacity(0.1)
        cone_actor.GetProperty().SetColor(colour)
        
        return cone_actor, world_position

class PanelArray:
    """
    Constructs and manages an array of flat panels with X-ray emitters.
    
    The array can be configured in various layouts (standard grid, staggered, custom)
    and performs geometric validation including collision detection between panels
    and emitter coverage checks.
    
    Attributes
    ----------
    panel_names : list of str
        Standard panel position names in 3x3 grid layout.
    panel_actors : list of vtk.vtkActor
        VTK actors representing panel bodies.
    emitter_actors : list of list of vtk.vtkActor
        Nested list of VTK actors representing emitter cones for each panel.
    cone_positions : list of list of ndarray
        World coordinates of all emitters for each panel.
    centred_emitters : list of ndarray
        Local coordinates of emitters for each panel before transformation.
    panel_collision : int
        Flag indicating panel-to-panel collision (0=no collision, 1=collision).
    panelXRay_collision : int
        Flag indicating panel-to-emitter collision (0=no collision, 1=collision).
    outside_wafer : bool
        Flag indicating if emitters extend beyond wafer boundaries.
    contained_panel_dist : bool
        Flag indicating if panels are within acceptable distance constraints.
    """
    
    # Standard 3x3 panel position names
    PANEL_NAMES = [
        'top_left', 'top', 'top_right',
        'left', 'centre', 'right',
        'bottom_left', 'bottom', 'bottom_right'
    ]
    
    # Maximum allowable distance between panel centers (meters)
    MAX_PANEL_SEPARATION = 0.8
    
    def __init__(self):
        """Initialise empty panel array."""
        # Configuration parameters
        self.panel_pitch = None
        self.emitter_pitch = None
        self.panel_theta = None
        self.panel_alpha = None
        self.cone_alpha = None
        self.activate = None
        self.panel_setting = None
        self.panel_positions = None
        self.stagger = None
        self.cone_angle = None
        self.SID = None
        self.cone_resolution = None
        self.emission_length = None
        self.grid = None
        self.silence_tqdm = None
        self.wafer_size = None
        
        # VTK objects
        self.panel_actors = []
        self.emitter_actors = []
        self.cone_positions = []
        self.centred_emitters = []
        
        # Validation flags
        self.panel_collision = 0
        self.panelXRay_collision = 0
        self.outside_wafer = False
        self.contained_panel_dist = True
        
        # Standard panel names
        self.panel_names = self.PANEL_NAMES
    
    def Construct(self, 
                  cone_angle=12.0,
                  panel_pitch=0.5,
                  emitter_pitch=0.025,
                  panel_theta=None,
                  panel_alpha=0.0,
                  cone_alpha=0.0,
                  activate=None,
                  panel_setting='standard grid',
                  panel_positions=None,
                  stagger=None,
                  SID=1.0,
                  cone_resolution=4,
                  emission_length=2.0,
                  grid=3,
                  silence_tqdm=True):
        """
        Construct the panel array with specified configuration.
        
        Parameters
        ----------
        cone_angle : float, optional
            X-ray cone emission angle in degrees. Default is 12.0.
        panel_pitch : float, optional
            Spacing between adjacent panels in meters. Default is 0.5.
        emitter_pitch : float, optional
            Spacing between emitters within a panel in meters. Default is 0.025.
        panel_theta : float, optional
            Panel rotation angle in degrees. If None, defaults to 2 * cone_angle.
        panel_alpha : float, optional
            Panel rotation around z-axis in degrees. Default is 0.0.
        cone_alpha : float, optional
            Cone rotation around z-axis in degrees. Default is 0.0.
        activate : ndarray, optional
            Binary array indicating which panels to activate (1=active, 0=inactive).
            Length must be 9. If None, all panels are activated.
        panel_setting : str, optional
            Panel layout configuration. Options: 'standard grid', 'staggered', 'custom'.
            Default is 'standard grid'.
        panel_positions : ndarray, optional
            Custom panel positions, shape (9, 3). Required if panel_setting='custom'.
        stagger : float, optional
            Z-offset for center panel in meters. Required if panel_setting='staggered'.
        SID : float, optional
            Source-to-image distance in meters. Default is 1.0.
        cone_resolution : int, optional
            Number of sides for cone mesh representation. Default is 4.
        emission_length : float, optional
            Height/length of emission cone in meters. Default is 2.0.
        grid : int, optional
            Grid size (grid x grid emitters per panel). Default is 3.
        silence_tqdm : bool, optional
            If True, suppress warning messages. Default is True.
        
        Raises
        ------
        AssertionError
            If stagger is not provided when panel_setting='staggered'.
        ValueError
            If panel_setting is not one of the valid options.
        """
        # Set default panel theta if not provided
        if panel_theta is None:
            panel_theta = cone_angle * 2
        
        # Set default activation (all panels active)
        if activate is None:
            activate = np.ones(9)
        
        # Store configuration
        self.cone_angle = cone_angle
        self.panel_pitch = panel_pitch
        self.emitter_pitch = emitter_pitch
        self.panel_theta = panel_theta
        self.panel_alpha = panel_alpha
        self.cone_alpha = cone_alpha
        self.activate = activate
        self.panel_setting = panel_setting
        self.panel_positions = panel_positions
        self.stagger = stagger
        self.cone_resolution = cone_resolution
        self.emission_length = emission_length
        self.SID = SID
        self.grid = grid
        self.silence_tqdm = silence_tqdm
        self.wafer_size = 0.1
        
        # Generate panel positions based on configuration
        positions = self._generate_panel_positions()
        
        # Filter for active panels only
        active_positions = positions[activate == 1]
        active_panel_names = np.array(self.panel_names)[activate == 1]
        
        # Construct individual panels
        self._construct_panels(active_panel_names, active_positions)
        
        # Perform geometric validation
        self._validate_geometry(active_panel_names)
    
    def _generate_panel_positions(self):
        """
        Generate panel positions based on panel_setting configuration.
        
        Returns
        -------
        ndarray
            Array of panel positions, shape (9, 3).
        
        Raises
        ------
        AssertionError
            If stagger not provided when panel_setting='staggered'.
        ValueError
            If panel_setting is invalid.
        """
        pp = self.panel_pitch
        sid = self.SID
        
        if self.panel_setting == 'standard grid':
            positions = np.array([
                (-pp, pp, sid),   (0, pp, sid),   (pp, pp, sid),
                (-pp, 0, sid),    (0, 0, sid),    (pp, 0, sid),
                (-pp, -pp, sid),  (0, -pp, sid),  (pp, -pp, sid)
            ])
        
        elif self.panel_setting == 'staggered':
            assert self.stagger is not None, (
                "panel_setting='staggered' requires 'stagger' parameter to be defined"
            )
            positions = np.array([
                (-pp, pp, sid),   (0, pp, sid),              (pp, pp, sid),
                (-pp, 0, sid),    (0, 0, sid + self.stagger), (pp, 0, sid),
                (-pp, -pp, sid),  (0, -pp, sid),             (pp, -pp, sid)
            ])
        
        elif self.panel_setting == 'custom':
            if self.panel_positions is None:
                if not self.silence_tqdm:
                    print('No custom panel positions provided. Reverting to standard grid.')
                positions = np.array([
                    (-pp, pp, sid),   (0, pp, sid),   (pp, pp, sid),
                    (-pp, 0, sid),    (0, 0, sid),    (pp, 0, sid),
                    (-pp, -pp, sid),  (0, -pp, sid),  (pp, -pp, sid)
                ])
            else:
                positions = self.panel_positions
        
        else:
            raise ValueError(
                f"Invalid panel_setting: '{self.panel_setting}'. "
                f"Must be one of: 'standard grid', 'staggered', or 'custom'"
            )
        
        return positions
    
    def _construct_panels(self, panel_names, positions):
        """
        Construct individual panel objects and collect their actors.
        
        Parameters
        ----------
        panel_names : ndarray
            Names of active panels.
        positions : ndarray
            Positions of active panels, shape (N, 3).
        """
        panel_actors = []
        emitter_actors = []
        cone_positions = []
        centred_emitters = []
        
        for i, (name, pos) in enumerate(zip(panel_names, positions)):
            # Create single panel
            panel = SinglePanel(
                SID=self.SID,
                emission_length=self.emission_length,
                emitter_pitch=self.emitter_pitch,
                centre=tuple(pos),
                cone_resolution=self.cone_resolution,
                panel_name=name,
                panel_theta=self.panel_theta,
                panel_alpha=self.panel_alpha,
                cone_alpha=self.cone_alpha,
                cone_angle=self.cone_angle,
                grid=self.grid,
                silence_tqdm=self.silence_tqdm
            )
            
            # Construct panel geometry
            panel.ConstructPanelBody(name=f'panel-{i}/panel-body')
            panel.EmitterArray(panel_number=i)
            
            # Collect actors and positions
            panel_actors.append(panel.cuboid_actor)
            emitter_actors.append(panel.cone_actors)
            cone_positions.append(panel.cone_positions)
            centred_emitters.append(panel.centred_emitters)
        
        # Store results
        self.panel_actors = panel_actors
        self.emitter_actors = emitter_actors
        self.cone_positions = cone_positions
        self.centred_emitters = centred_emitters
    
    def _validate_geometry(self, panel_names):
        """
        Perform geometric validation on the constructed panel array.
        
        Checks for:
        - Emitters extending beyond wafer boundaries
        - Panel-to-panel collisions
        - Panel-to-emitter collisions
        - Panel separation constraints
        
        Parameters
        ----------
        panel_names : ndarray
            Names of active panels.
        """
        self._check_emitters_in_wafer()
        self._check_panel_collisions()
        self._check_panel_emitter_collisions(panel_names)
        self._check_panel_separation()
    
    def _check_emitters_in_wafer(self):
        """
        Check if all emitters fall within the wafer boundaries.
        
        Sets self.outside_wafer flag.
        """
        if len(self.centred_emitters) == 0:
            self.outside_wafer = False
            return
        
        # Check first panel's emitters (representative of all panels)
        emitters = self.centred_emitters[0]
        emitter_x = emitters[:, 0]
        emitter_y = emitters[:, 1]
        
        # Check if any emitters are outside circular wafer boundary
        wafer_radius = self.wafer_size / 2.0
        inside_wafer = (emitter_x**2 + emitter_y**2) <= wafer_radius**2
        any_outside = np.sum(~inside_wafer) > 0
        
        self.outside_wafer = any_outside
        
        if self.outside_wafer and not self.silence_tqdm:
            print("WARNING: Emitters extend beyond wafer boundaries.")
    
    def _check_panel_collisions(self):
        """
        Check for collisions between panel bodies.
        
        Sets self.panel_collision flag.
        """
        if len(self.panel_actors) < 2:
            self.panel_collision = 0
            return
        
        # Check all panel pairs for collisions
        collision_detected = False
        for i in range(len(self.panel_actors)):
            if collision_detected:
                break
            
            panel_i = self.panel_actors[i]
            
            for j in range(i + 1, len(self.panel_actors)):
                panel_j = self.panel_actors[j]
                
                # Quick bounding box check first
                if clipping_utils.CheckCollisionBoundingBox(panel_i, panel_j) == 0:
                    continue
                
                # Detailed collision check
                if clipping_utils.CheckCollision(panel_i, panel_j) == 1:
                    collision_detected = True
                    break
        
        self.panel_collision = 1 if collision_detected else 0
        
        if self.panel_collision == 1 and not self.silence_tqdm:
            print("WARNING: Panel collision detected.")
    
    def _check_panel_emitter_collisions(self, panel_names):
        """
        Check for collisions between panels and emitters from center panel.
        Note that this method will fail if using custom panel positions wherein
        edge or corner panels may be irradiating other panels.
        
        Parameters
        ----------
        panel_names : ndarray
            Names of active panels.
        
        Sets self.panelXRay_collision flag.
        """
        # Find center panel index
        panel_indices = np.arange(len(panel_names))
        center_indices = panel_indices[panel_names == 'centre']
        
        if len(center_indices) == 0:
            # No center panel active
            self.panelXRay_collision = 0
            return
        
        center_idx = center_indices[0]
        collision_detected = False
        
        # Check each non-center panel against center panel's emitters
        for i in range(len(self.panel_actors)):
            if i == center_idx or collision_detected:
                continue
            
            panel = self.panel_actors[i]
            
            # Check against each emitter from center panel
            for emitter in self.emitter_actors[center_idx]:
                # Quick bounding box check
                if clipping_utils.CheckCollisionBoundingBox(panel, emitter) == 0:
                    continue
                
                # Detailed collision check
                if clipping_utils.CheckPanelXRayCollision(panel, emitter) == 1:
                    collision_detected = True
                    break
        
        self.panelXRay_collision = 1 if collision_detected else 0
        
        if self.panelXRay_collision == 1 and not self.silence_tqdm:
            print("WARNING: Panel obstructs X-ray emission from center panel.")
    
    def _check_panel_separation(self):
        """
        Check if panel separation distances are within acceptable limits.
        
        Sets self.contained_panel_dist flag.
        """
        if len(self.panel_actors) < 2:
            self.contained_panel_dist = True
            return
        
        # Extract panel center positions
        panel_centers = []
        for actor in self.panel_actors:
            xmin, xmax, ymin, ymax, zmin, zmax = actor.GetBounds()
            center_x = (xmin + xmax) / 2.0
            center_y = (ymin + ymax) / 2.0
            panel_centers.append([center_x, center_y])
        
        panel_centers = np.array(panel_centers)
        
        # Calculate all pairwise distances
        distances = []
        for i in range(len(panel_centers) - 1):
            pos_i = panel_centers[i]
            for j in range(i + 1, len(panel_centers)):
                pos_j = panel_centers[j]
                distance = np.linalg.norm(pos_j - pos_i)
                distances.append(distance)
        
        # Check maximum distance
        if len(distances) > 0:
            max_distance = np.nanmax(distances)
            self.contained_panel_dist = max_distance <= self.MAX_PANEL_SEPARATION
        else:
            self.contained_panel_dist = True
    
    def get_validation_summary(self):
        """
        Get a summary of geometric validation results.
        
        Returns
        -------
        dict
            Dictionary containing validation flags and their descriptions.
        """
        return {
            'outside_wafer': {
                'flag': self.outside_wafer,
                'description': 'Emitters extend beyond wafer boundaries'
            },
            'panel_collision': {
                'flag': bool(self.panel_collision),
                'description': 'Panel bodies collide with each other'
            },
            'panelXRay_collision': {
                'flag': bool(self.panelXRay_collision),
                'description': 'Panels obstruct X-ray emission paths'
            },
            'contained_panel_dist': {
                'flag': self.contained_panel_dist,
                'description': f'All panels within {self.MAX_PANEL_SEPARATION}m separation'
            }
        }
    
    def is_valid_geometry(self):
        """
        Check if the panel array has valid geometry.
        
        Returns
        -------
        bool
            True if geometry passes all validation checks, False otherwise.
        """
        return (not self.outside_wafer and
                self.panel_collision == 0 and
                self.panelXRay_collision == 0 and
                self.contained_panel_dist)