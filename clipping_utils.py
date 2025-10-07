import vtk
from vtk.util.colors import peacock, tomato
import numpy as np
from math import sqrt as sqrt
from vtk.util import numpy_support
import interactor_utils
from tqdm import tqdm
import aux_material

def extract_triangles(actor):
    actual_points = actor.GetMapper().GetInput().GetPoints().GetData()
    actual_points = numpy_support.vtk_to_numpy(actual_points)

    actual_indices = numpy_support.vtk_to_numpy(actor.GetMapper().GetInput().GetPolys().GetData()) # array is typically 1D with the number of vertices included.
    actual_indices = actual_indices.reshape([-1,4]) # reshapes to a n*4 with (hopefully) the first collumn being 3s 

    assert np.all(actual_indices[:, 0] == 3), "mesh cells contain non-triangles, try using vtkTriangleFilter first"
    
    actual_indices = actual_indices[:,1:] # this removes the first column

    return actual_points, actual_indices

def clipper(actor_to_clip, clipping_function, fill=False, opacity=0.1, name='intersection'):

    # used to clip polydata with an implicit function like a plane, cylinder etc. 

    cutEdges = vtk.vtkCutter()
    cutEdges.SetInputData(actor_to_clip.GetMapper().GetInput())
    cutEdges.SetCutFunction(clipping_function)
    cutEdges.GenerateCutScalarsOn()
    cutEdges.SetValue(0, 0.5)
    cutStrips = vtk.vtkStripper()
    cutStrips.SetInputConnection(cutEdges.GetOutputPort())
    cutStrips.Update()
    cutPoly = vtk.vtkPolyData()
    cutPoly.SetPoints(cutStrips.GetOutput().GetPoints())
    cutPoly.SetPolys(cutStrips.GetOutput().GetLines())


    total_cut_edges = cutEdges.GetOutput().GetNumberOfPoints()
    # print('Total cut edges %i' % total_cut_edges)
    cutterMapper = vtk.vtkPolyDataMapper()
    cutterMapper.SetInputData(cutEdges.GetOutput())
    cutterActor = vtk.vtkActor()
    cutterActor.SetMapper(cutterMapper)
    cutterActor.GetProperty().SetColor( 1, 0, 0 )
    cutterActor.GetProperty().SetLineWidth(2)
    cutterActor.SetObjectName(name)

    if fill:

        cutTriangles = vtk.vtkTriangleFilter()
        cutTriangles.SetInputData(cutPoly)
        cutTriangles.Update()
        cutMapper = vtk.vtkPolyDataMapper()
        cutMapper.SetInputData(cutPoly)
        cutMapper.SetInputConnection(cutTriangles.GetOutputPort())
        cutActor = vtk.vtkActor()
        cutActor.SetMapper(cutMapper)
        cutActor.GetProperty().SetColor(0, 1, 1)
        cutActor.GetProperty().SetLineWidth(2)
        cutActor.GetProperty().SetOpacity(opacity)
        cutMapper.Update()
        cutActor.SetObjectName(name)
        cutActor.Modified()

        # extract_triangles(cutActor)

        return cutActor

    else:

        return cutterActor

def transform_PolyData(actor):
    """
    Function obtains clean polydata from actor in 3 steps 
    1. get polydata from actor : source: https://stackoverflow.com/questions/35956983/updating-vtkpolydata-from-vtkactor 
    2. triangulation - converts input polygons and strips to triangles) 
    3. cleaning - merges duplicate points (within specified tolerance and if enabled), 
       eliminates points that are not used in any cell, and if enabled, 
       transforms degenerate cells into appropriate forms   (for example, a triangle 
       is converted into a line if two points of triangle are merged).
    """
    # 1. Copy the input polydata from the actor and apply a transformation from the actor's matrix
    polyData = vtk.vtkPolyData()
    polyData.DeepCopy(actor.GetMapper().GetInput())
    
    transform = vtk.vtkTransform()
    transform.SetMatrix(actor.GetMatrix())
    fil = vtk.vtkTransformPolyDataFilter()
    fil.SetTransform(transform)
    fil.SetInputDataObject(polyData)
    fil.Update()
    polyData.DeepCopy(fil.GetOutput())
    
    # 2. Triangulate the polydata
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(polyData)
    triangle_filter.Update()
    
    # 3. Clean the polydata
    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.SetInputData(triangle_filter.GetOutput())
    clean_filter.Update()
    return clean_filter.GetOutput(); 

def CheckCollision(Actor1, Actor2, timeout=5):

    """
    Function checks whether two actors intersect with each other using vtk Boolean Operations
    """
    def perturb_actor(actor, max_translation=0.001, max_rotation=0.001):
        # Apply a small random translation
        translation = np.random.uniform(-max_translation, max_translation, 3)
        actor.SetPosition(np.array(actor.GetPosition()) + translation)

        # Apply a small random rotation
        rotation = np.random.uniform(-max_rotation, max_rotation, 3)
        actor.RotateX(rotation[0])
        actor.RotateY(rotation[1])
        actor.RotateZ(rotation[2])

        return actor


    #Actor1 = perturb_actor(Actor1)
    #Actor2 = perturb_actor(Actor2)

    input1 = transform_PolyData(Actor1)
    input2 = transform_PolyData(Actor2)

    def check_polydata_validity(polydata):
        if polydata is None:
            print("Polydata is None.")
            return False

        if polydata.GetNumberOfPoints() == 0 or polydata.GetNumberOfCells() == 0:
            print("Polydata has no points or cells.")
            return False

        return True

    # Check the validity of input polydata before applying the clean filter
    if not check_polydata_validity(input1):
        print("Input1 is invalid.")
    if not check_polydata_validity(input2):
        print("Input2 is invalid.")


    def save_polydata(polydata, filename):
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(polydata)
        writer.Write()

    # Save the input polydata to .vtp files
    #save_polydata(input1, "input1.vtp")
    #save_polydata(input2, "input2.vtp")

    def distance_between_points(p1, p2):
        return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    def check_points_between_polydata(polydata1, polydata2, threshold):
        points1 = polydata1.GetPoints()
        points2 = polydata2.GetPoints()
        
        num_points1 = points1.GetNumberOfPoints()
        num_points2 = points2.GetNumberOfPoints()
        
        close_point_pairs = []

        for i in range(num_points1):
            p1 = points1.GetPoint(i)
            for j in range(num_points2):
                p2 = points2.GetPoint(j)
                distance = distance_between_points(p1, p2)
                if distance < threshold:
                    close_point_pairs.append((i, j, distance))
        
        return close_point_pairs

    threshold = 0.02 

    # First checks if there are any points very close together. This can result in VTK hanging, so it is a useful first step.   
    close_point_pairs = check_points_between_polydata(input1, input2, threshold)

    if len(close_point_pairs) > 0:
        #print('Too close!!')
        collision = 1
    else:
        booleanOperation = vtk.vtkBooleanOperationPolyDataFilter()
        booleanOperation.SetOperationToIntersection()

        booleanOperation.SetInputData(0, input1)

        booleanOperation.SetInputData(1, input2)
        booleanOperation.Update()

        booleanOperationMapper = vtk.vtkPolyDataMapper()
        booleanOperationMapper.SetInputConnection(booleanOperation.GetOutputPort())
        booleanOperationMapper.ScalarVisibilityOff()
        booleanOperationMapper.Update()
        
        if booleanOperationMapper.GetInput().GetNumberOfPoints() != 0:
            collision = 1
        else:
            collision = 0

    return collision

def CheckPanelXRayCollision(cubeActor, coneActor):
    """
    Function checks whether point of first actor is inside of the closed surface mesh, 
    formed by second actor, using vtkSelectEnclosedPoints method. 
    """
    # Clean polydata from actor: 
    cubeinput = transform_PolyData(cubeActor)
    coneinput = transform_PolyData(coneActor)   
    
    # Create a vtkSelectEnclosedPoints object to perform point-inclusion check:
    selectEnclosedPoints = vtk.vtkSelectEnclosedPoints()        
    #selectEnclosedPoints.SetTolerance(0.0000000000001)  # might not needed
    
    # Set the input data (cube) and the surface data (cone) for the inclusion check:     
    selectEnclosedPoints.SetInputData(cubeinput)
    selectEnclosedPoints.SetSurfaceData(coneinput)

    selectEnclosedPoints.Update()
    
    
    # Check if any point is inside the closed surface, if yes then panel and XRay cone collide:        
    inside = False
    for i in range (cubeinput.GetNumberOfPoints()):
        if selectEnclosedPoints.IsInsideSurface(cubeinput.GetPoint(i)) == 1:
            inside = True
            break

    if inside:
        panelXRay_collision=1
    else:
        panelXRay_collision=0
    
    return panelXRay_collision   
    
def CheckCollisionBoundingBox(actor1, actor2):
   bounds_i = actor1.GetBounds()
   bounds_j = actor2.GetBounds()
   if bounds_i[1] > bounds_j[0] and bounds_i[0] < bounds_j[1] and bounds_i[3] > bounds_j[2] and bounds_i[2] < bounds_j[3]:
       collision = 1
   else:
       collision=0
   return collision    