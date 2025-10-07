import numpy as np
from tqdm import tqdm
import os
import vtk
import warnings
import PIL
from vtk.util import numpy_support

def UVTcoords(uResolution, vResolution, pd):
    """
    Generate u, v texture coordinates on a parametric surface.
    :param uResolution: u resolution
    :param vResolution: v resolution
    :param pd: The polydata representing the surface.
    :return: The polydata with the texture coordinates added.
    """
    print('Can\'t find texture coordinates, making new ones')
    numPts = pd.GetNumberOfPoints()

    if ((uResolution is None) or (vResolution is None)):
        limit = int(np.floor(np.sqrt(numPts)))
        uResolution = limit
        vResolution = limit

    elif numPts < (uResolution * vResolution):
        limit = int(np.floor(np.sqrt(numPts)))
        warnings.warn('Texture coords set too high, setting both to: %i' % limit)
        uResolution = limit
        vResolution = limit

    u0 = 1.0
    v0 = 0.0
    du = 1.0 / (uResolution - 1)
    dv = 1.0 / (vResolution - 1)

    tCoords = vtk.vtkFloatArray()
    tCoords.SetNumberOfComponents(2)
    tCoords.SetNumberOfTuples(numPts)
    tCoords.SetName('Texture Coordinates')
    ptId = 0
    u = u0
    for i in range(0, uResolution):
        v = v0
        for j in range(0, vResolution):
            tc = [u, v]
            # print(ptId, tc)
            tCoords.SetTuple(ptId, tc)
            v += dv
            ptId += 1
        u -= du
    pd.GetPointData().SetTCoords(tCoords)
    return pd

def read_directory(directory, actor_dict, mesh_color='blue', texture_name=None, use_textures=False, opacity=1.0, rotate=[0,0,0], translate=[0,0,0], scale=[1,1,1]):

    actor_dict_2 = dict()

    dirname = directory.split('/')[-1]

    # print(dirname)

    for filename in tqdm(os.listdir(directory), desc='Reading %s' % directory):
        filepath = directory + os.sep + filename 

        if filepath.endswith('.obj'):

            obj_actor = load_obj(filepath,
                                    scale=scale,
                                    translate=translate,
                                    rotate=rotate,
                                    opacity=opacity,
                                    mesh_color=mesh_color)

            if use_textures:

                obj_actor = add_textures_to_actor(obj_actor, texture_name, texture_dir=directory + os.sep)

            actor_dict_2.update({filename: obj_actor})
    
    actor_dict.update({dirname: actor_dict_2})

def read_texture(filename, file_type='jpg', useSRGB=False):

    if file_type == '.png':
        color = vtk.vtkPNGReader()
    if file_type == '.jpg':
        color = vtk.vtkJPEGReader()
    elif file_type == '.tiff':
        warnings.warn('TIFF images may produce odd effects, JPEGs are recommended')
        color = vtk.vtkTIFFReader()
    elif file_type == '.tif':
        warnings.warn('TIFF images may produce odd effects, JPEGs are recommended')
        color = vtk.vtkTIFFReader()
    color.SetFileName(filename)
    color_texture = vtk.vtkTexture()
    color_texture.SetInputConnection(color.GetOutputPort())

    if useSRGB:
        color_texture.UseSRGBColorSpaceOn()

    return color_texture

def read_texture_directory(folder_path, color=None, texture_size=[1024,1024], fallback_ORM=[128,0,0], fallback_normals=[128,128, 255], fallback_height=128):

    file_type = '.jpg'

    if color is None:

        albedofile = folder_path + os.sep + "albedo" + file_type

    else:

        albedofile = folder_path + os.sep + color + file_type


    if not os.path.isfile(albedofile):

        file_type = '.png'

        if color is None:

            albedofile = folder_path + os.sep + "albedo" + file_type

        else:

            albedofile = folder_path + os.sep + color + file_type

    if not os.path.isfile(albedofile):


        file_type = '.tif'

        if color is None:

            albedofile = folder_path + os.sep + "albedo" + file_type

        else:

            albedofile = folder_path + os.sep + color + file_type

    albedo_texture = read_texture(albedofile, useSRGB=True, file_type=file_type)


    normalfile = folder_path + os.sep + "normal" + file_type
    heightfile = folder_path + os.sep + "height" + file_type
    emissivefile = folder_path + os.sep + "emissive" + file_type
    ormfile =  folder_path + os.sep + "orm" + file_type

    a = np.array(PIL.Image.open(albedofile))
    if np.ndim(a) == 3:
        a = a[:,:,0]
    texture_size = a.shape

    if os.path.isfile(ormfile):

        orm_texture = read_texture(ormfile, file_type=file_type)

    else:

        print('Can\'t find ORM file, assuming seperate Occlusion, Roughness & Metalicity')

        occlusionfile = folder_path + os.sep + "ao" + file_type
        roughnessfile = folder_path + os.sep + "roughness" + file_type
        specularfile = folder_path + os.sep + "specular" + file_type
        metalfile = folder_path + os.sep + "metallic" + file_type

        if os.path.isfile(occlusionfile):
            o = np.array(PIL.Image.open(occlusionfile))
            if np.ndim(o) == 3:
                o = o[:,:,0]
        else:
            print('Occlusion file not found, using fallback value: %f' % fallback_ORM[0])
            o = np.ones(texture_size) * fallback_ORM[0]

        if os.path.isfile(roughnessfile):
            r = np.array(PIL.Image.open(roughnessfile))
            if np.ndim(r) == 3:
                r = r[:,:,0]
        elif os.path.isfile(specularfile):
            r = 1 - np.array(PIL.Image.open(specularfile))
            if np.ndim(r) == 3:
                r = r[:,:,0]
                print(r)
        else:
            print('Roughness file not found, using fallback value: %f' % fallback_ORM[1])
            r = np.ones(texture_size) * fallback_ORM[1]

        if os.path.isfile(metalfile):
            m = np.array(PIL.Image.open(metalfile))
            if np.ndim(m) == 3:
                m = m[:,:,0]
        else:
            print('Metalicity file not found, using fallback value: %f' % fallback_ORM[2])
            m = np.ones(texture_size) * fallback_ORM[2]

        orm = np.dstack([o,r,m])

        orm_image = PIL.Image.fromarray(orm.astype(np.uint8))
        print('saving new ORM texture to %s' % ormfile)
        orm_image.save(ormfile)

        orm_texture = read_texture(ormfile, file_type=file_type)


    if os.path.isfile(normalfile):
        normal_texture = read_texture(normalfile, file_type=file_type)
    else:
        print('Normal file not found, creating fallback')
        normal_array = np.ones([*a.shape, 3]) * fallback_normals
        normal_image = PIL.Image.fromarray(normal_array.astype(np.uint8))
        normal_image.save(normalfile)
        normal_texture = read_texture(normalfile, file_type=file_type)


    texture_dict = dict()

    texture_dict.update({'albedo':albedo_texture})
    texture_dict.update({'normal':normal_texture})
    texture_dict.update({'ORM':orm_texture})

    if os.path.isfile(heightfile):
        height_texture = read_texture(heightfile, useSRGB=True, file_type=file_type)
        texture_dict.update({'height':height_texture})
    else:
        pass

    if os.path.isfile(emissivefile):
        emissive_texture = read_texture(emissivefile, useSRGB=True, file_type=file_type)
        texture_dict.update({'emissive':emissive_texture})
    else:
        pass

    return texture_dict

def add_textures_to_actor(actor, material=None, texture_dir='data/textures/', color=None, occlusion=0.5, roughness=1, metallic=1, normal_scale=1, height_scale=0.01, emissive=(1,1,1), uResolution=None, vResolution=None, use_heightmap=False, method='vtk', texture_scale=(1,1,1), verbose=False):

    fallback_ORM = np.asarray([occlusion, roughness, metallic]) * 255

    if material is not None:
        texture_dict = read_texture_directory(texture_dir + material, fallback_ORM=fallback_ORM, color=color)
    else:
        texture_dict = read_texture_directory(texture_dir, fallback_ORM=fallback_ORM, color=color)

    if (('height' in texture_dict) and (use_heightmap)):

        if material is not None:
            filepath = texture_dir + material + '/height.jpg'
        else:
            filepath = texture_dir + 'height.jpg'
        reader = vtk.vtkJPEGReader()
        if not os.path.isfile(filepath):
            if material is not None:
                filepath = texture_dir + material + '/height.png'
            else:
                filepath = texture_dir + 'height.png'
            reader = vtk.vtkPNGReader()
            if not os.path.isfile(filepath):
                if material is not None:
                    filepath = texture_dir + material + '/height.tiff'
                else:
                    filepath = texture_dir + 'height.tiff'
                reader = vtk.vtkTIFFReader()

        if verbose:

            print('Height displacement enabled, using %s' % filepath)


        reader.SetFileName(filepath)
        reader.Update()

        image_extents = np.asarray(reader.GetDataExtent())
        image_size = np.asarray([image_extents[1] - image_extents[0], image_extents[3] - image_extents[2], image_extents[5] - image_extents[4]])



        current_tocoords = actor.GetMapper().GetInput().GetPointData().GetTCoords()

        # print('Current Coordinates', current_tocoords)

        if current_tocoords is not None:

            mapper = actor.GetMapper()

            probe_points = vtk.vtkPoints()

            # print(current_tocoords)

            # print(image_size)

            probe_points.SetNumberOfPoints(current_tocoords.GetNumberOfTuples())

            # print(probe_points.GetDataType())

            # print(dir(probe_points))

            np_coords = numpy_support.vtk_to_numpy(current_tocoords)
            n_0 = np.zeros([current_tocoords.GetNumberOfTuples(),1])
            
            new_coords = numpy_support.numpy_to_vtk(np.hstack([np_coords, n_0])  * image_size)

            # probe_points.SetNumberOfComponents(new_coords.GetNumberOfComponents())
            probe_points.SetData(new_coords)
            probe_poly = vtk.vtkPolyData()
            probe_poly.SetPoints(probe_points)

        else:

            pd = UVTcoords(uResolution, vResolution, actor.GetMapper().GetInput())

            tcoords = pd.GetPointData().GetTCoords()

            probe_points = vtk.vtkPoints()
            probe_points.SetNumberOfPoints(tcoords.GetNumberOfValues())

            np_coords = numpy_support.vtk_to_numpy(tcoords)
            n_0 = np.zeros([tcoords.GetNumberOfTuples(),1])

            probe_points.SetData(numpy_support.numpy_to_vtk(np.hstack([np_coords, n_0]) * image_size))

            probe_poly = vtk.vtkPolyData()
            probe_poly.SetPoints(probe_points)

            current_tocoords = probe_poly

        probes = vtk.vtkProbeFilter()
        probes.SetSourceData(reader.GetOutput())
        probes.SetInputData(probe_poly)
        probes.Update()

        actor.GetMapper().GetInput().GetPointData().SetScalars(probes.GetOutput().GetPointData().GetScalars())

        warp = vtk.vtkWarpScalar()
        warp.SetInputData(actor.GetMapper().GetInput())
        warp.SetScaleFactor(height_scale)
        warp.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(warp.GetOutputPort())
        mapper.GetInput().GetPointData().SetScalars(None)

        smoothing_passes = 1

        if smoothing_passes is not None:

            if verbose:
                print('Smoothing mesh, may take a while')

            smooth_loop = vtk.vtkSmoothPolyDataFilter()
            smooth_loop.SetNumberOfIterations(smoothing_passes)
            smooth_loop.SetRelaxationFactor(0.5)
            smooth_loop.BoundarySmoothingOn()
            smooth_loop.SetInputData(mapper.GetInput())
            smooth_loop.Update()
            mapper = vtk.vtkPolyDataMapper()

            mapper.SetInputConnection(smooth_loop.GetOutputPort())

        # tcoords_np = numpy_support.vtk_to_numpy(current_tocoords)

        # print(tcoords_np.shape)

        # plt.figure()
        # plt.scatter(tcoords_np[:,0], tcoords_np[:,1])
        # plt.show()


    else:

        # print(actor)

        current_tocoords = actor.GetMapper().GetInput().GetPointData().GetTCoords()



        # tcoords_np = numpy_support.vtk_to_numpy(current_tocoords)

        # print(tcoords_np)

        # plt.figure()
        # plt.scatter(tcoords_np[:,0], tcoords_np[:,1])
        # plt.show()

        # print('current texture coordinates', current_tocoords)

        if current_tocoords is None:

            pd = UVTcoords(uResolution, vResolution, actor.GetMapper().GetInput())

            tangents = vtk.vtkPolyDataTangents()
            tangents.SetComputePointTangents(True)
            tangents.SetComputeCellTangents(True)
            tangents.SetInputData(pd)
            tangents.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(tangents.GetOutputPort())

        else: 

            mapper = actor.GetMapper()

    actor = vtk.vtkActor()

    actor.SetMapper(mapper)

    actor.GetProperty().SetRoughness(roughness)
    actor.GetProperty().SetMetallic(metallic)
    actor.GetProperty().SetOcclusionStrength(occlusion)
    actor.GetProperty().SetNormalScale(normal_scale)
    actor.GetProperty().SetEmissiveFactor(emissive)


    actor.GetProperty().SetInterpolationToPBR()
    actor.SetTexture(texture_dict['albedo'])
    actor.GetProperty().SetBaseColorTexture(texture_dict['albedo'])
    actor.GetProperty().SetNormalTexture(texture_dict['normal'])
    actor.GetProperty().SetORMTexture(texture_dict['ORM'])

    if 'emissive' in texture_dict:

        actor.GetProperty().SetEmissiveTexture(texture_dict['emissive'])

    return actor

def load_obj(filename,
             mtl_filename=None,
             renderer=None,
             opacity=1.0,
             specular=0.1,
             ambient=0.0,
             scale=(1, 1, 1),
             translate=(0, 0, 0),
             rotate=(0, 0, 0),
             mesh_color="blue",
             use_wireframe=False,
             scale_then_translate=False,
             smoothing_passes=None,
             max_subdivisions=0,
             subdivision_method='adaptive',
             verbose=False
             ):
    """[Used to load in standard OBJ files and translate them]

    Arguments:
        filename {[string]} -- [the filename of the OBJ]

    Keyword Arguments:
        opacity {float} -- [opacity of the object] (default: {1.0})
        scale {tuple} -- [x,y,z scale] (default: {(1, 1, 1)})
        translate {tuple} -- [x,y,z shift] (default: {(0, 0, 0)})
        rotate {tuple} -- [rotation in degrees] (default: {(0, 0, 0)})
        mesh_color {str} -- [color of the mesh, taken from VTK colors] (default: {"blue"})
        use_wireframe {bool} -- [renders the wireframe instead] (default: {False})
    """
    # print('---------------------------------------------------', filename)
    colors = vtk.vtkNamedColors()


    if mtl_filename is not None:
        reader = vtk.vtkOBJImporter()
        reader.SetFileName(filename)
        reader.SetFileNameMTL(mtl_filename)
        reader.InitializeObjectBase()

        mtl_foldername = os.path.dirname(mtl_filename)

        # print(mtl_foldername)

        reader.SetTexturePath(mtl_foldername)

        reader.Modified()
        reader.Read()

        renderer = reader.GetRenderer()

        # print('Importer dir', dir(reader))

        # print(reader.GetTexturePath())

        actors = renderer.GetActors()

        actors.InitTraversal()

        # print(dir(actors))

        # print('total items', actors.GetNumberOfItems())

        actor_dict = dict()
        actor_list = list()
        iterator = 0

        for i in range(actors.GetNumberOfItems()+1):

            # print(i)

            # print(actors.GetLastActor())
            
            actor = actors.GetItemAsObject(i)
            # actor = actors.GetLastActor()

            if actor is None:
                break

            number_of_points = actor.GetMapper().GetInput().GetNumberOfPoints()

            if number_of_points == 0:
                print('Number of points read is zero, assuming this is empty, skipping')
                continue

            # print('Number of points', )

            # print(actor)

            # actor.GetProperty().SetColor(colors.GetColor3d(mesh_color))
            actor.GetProperty().SetOpacity(opacity)
            actor.GetProperty().SetSpecular(specular)
            actor.GetProperty().SetSpecularPower(1.0)
            actor.GetProperty().SetAmbient(ambient)





            transform = vtk.vtkTransform()
            transform.RotateX(rotate[0])
            transform.RotateY(rotate[1])
            transform.RotateZ(rotate[2])

            if scale_then_translate:
                transform.Scale(scale)
                transform.Translate(translate)
            else:
                transform.Translate(translate)
                transform.Scale(scale)

            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetInputData(actor.GetMapper().GetInput())
            transformFilter.SetTransform(transform)
            transformFilter.Update()

            if max_subdivisions > 0:

                if verbose:
                    print('Subdividing mesh, may take a while, limit of', max_subdivisions)

                if subdivision_method == 'adaptive':
                    subdivider = vtk.vtkAdaptiveSubdivisionFilter()
                    subdivider.SetMaximumNumberOfPasses(max_subdivisions)
                elif subdivision_method == 'linear':
                    subdivider = vtk.vtkLinearSubdivisionFilter()
                    subdivider.SetNumberOfSubdivisions(max_subdivisions)
                elif subdivision_method == 'butterfly':
                    subdivider = vtk.vtkButterflySubdivisionFilter()
                    subdivider.SetNumberOfSubdivisions(max_subdivisions)

                subdivider.SetInputConnection(transformFilter.GetOutputPort())
                subdivider.Update()
                transformFilter = subdivider

                if verbose:
                    print('Mesh sibdivided')



            if smoothing_passes is not None:

                if verbose:
                    print('Smoothing mesh, may take a while')

                smooth_loop = vtk.vtkSmoothPolyDataFilter()
                smooth_loop.SetNumberOfIterations(smoothing_passes)
                smooth_loop.SetRelaxationFactor(0.5)
                smooth_loop.BoundarySmoothingOn()
                smooth_loop.SetInputConnection(transformFilter.GetOutputPort())
                smooth_loop.Update()
                mapper = vtk.vtkPolyDataMapper()

                mapper.SetInputConnection(smooth_loop.GetOutputPort())
            else:
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(transformFilter.GetOutputPort())

            triangulation = vtk.vtkTriangleFilter()
            triangulation.SetInputData(transformFilter.GetOutput())
            triangulation.Update()
            mapper.SetInputConnection(triangulation.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            if mtl_filename is None:
                actor.GetProperty().SetColor(colors.GetColor3d(mesh_color))
            actor.GetProperty().SetOpacity(opacity)

            # print(actor.GetProperty().GetSpecular())
            # print(actor.GetProperty().GetAmbient())

            actor.GetProperty().SetSpecular(specular)
            actor.GetProperty().SetSpecularPower(80.0)
            actor.GetProperty().SetAmbient(ambient)
            actor.SetObjectName(filename)

            # actor.GetProperty().SetInterpolationToGouraud()

            actor_list.append(actor)

            actor_dict.update({str(iterator): actor_list[iterator]})

            iterator += 1
        # print(actor)

        return actor_dict

    else:    
        reader = vtk.vtkOBJReader()
        reader.SetFileName(filename)
        reader.Update()

    transform = vtk.vtkTransform()
    transform.RotateX(rotate[0])
    transform.RotateY(rotate[1])
    transform.RotateZ(rotate[2])

    if scale_then_translate:
        transform.Scale(scale)
        transform.Translate(translate)
    else:
        transform.Translate(translate)
        transform.Scale(scale)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputConnection(reader.GetOutputPort())
    transformFilter.SetTransform(transform)
    transformFilter.Update()

    if max_subdivisions > 0:

        print('subdividing mesh, may take a while')

        subdivider = vtk.vtkAdaptiveSubdivisionFilter()
        subdivider.SetMaximumNumberOfPasses(max_subdivisions)
        subdivider.SetInputConnection(transformFilter.GetOutputPort())
        subdivider.Update()
        transformFilter = subdivider



    if smoothing_passes is not None:

        smooth_loop = vtk.vtkSmoothPolyDataFilter()
        smooth_loop.SetNumberOfIterations(smoothing_passes)
        smooth_loop.SetRelaxationFactor(0.1)
        smooth_loop.BoundarySmoothingOn()
        smooth_loop.SetInputConnection(transformFilter.GetOutputPort())
        smooth_loop.Update()
        mapper = vtk.vtkPolyDataMapper()

        mapper.SetInputConnection(smooth_loop.GetOutputPort())
    else:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(transformFilter.GetOutputPort())

    triangulation = vtk.vtkTriangleFilter()
    triangulation.SetInputData(transformFilter.GetOutput())
    triangulation.Update()

    # fillHolesFilter = vtk.vtkFillHolesFilter()
    # fillHolesFilter.SetInputData(triangulation.GetOutput())
    # fillHolesFilter.SetHoleSize(100000.0)
    # fillHolesFilter.Update()

    # mapper.SetInputConnection(fillHolesFilter.GetOutputPort())

    mapper.SetInputConnection(triangulation.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    if mtl_filename is None:
        actor.GetProperty().SetColor(colors.GetColor3d(mesh_color))
    actor.GetProperty().SetOpacity(opacity)

    # print(actor.GetProperty().GetSpecular())
    # print(actor.GetProperty().GetAmbient())

    actor.GetProperty().SetSpecular(specular)
    actor.GetProperty().SetSpecularPower(80.0)
    actor.GetProperty().SetAmbient(ambient)

    actor.GetProperty().SetInterpolationToGouraud()
    actor.SetObjectName(filename)

    # featureEdges = vtk.vtkFeatureEdges()
    # featureEdges.SetInputData(actor.GetMapper().GetInput())
    # featureEdges.FeatureEdgesOff()
    # featureEdges.BoundaryEdgesOn()
    # featureEdges.NonManifoldEdgesOn()

    # featureEdges.Update()
    
    # open_edges = featureEdges.GetOutput().GetNumberOfCells()
    # total_edges = actor.GetMapper().GetInput().GetNumberOfCells()

    # assert open_edges == 0, 'Mesh has %i open edges, %i total' % (open_edges, total_edges)

    if use_wireframe:
        actor.GetProperty().SetRepresentationToWireframe()

    if renderer is None:
        return actor
    else:
        renderer.AddActor(actor)

def read_body(scale = 1,
                translate=(1.5, 0, -0.5),
                rotate=(0, 0, 0),
                opacity=0.5):
    s = scale

    init_path = '/Users/alexhill/Documents/GitHub/SCIMITAR_PUBLIC/female/'

    actor_dict = dict()

    body_actor = load_obj(init_path+'body/Female_Body_Skin.obj',
                                    scale=(s, s, s),
                                    translate=translate,
                                    rotate=rotate,
                                    opacity=opacity,
                                    mesh_color='misty_rose')

    skeletal_dir = init_path+'skeletal'
    skull_dir = init_path+'skull'

    read_directory(skeletal_dir, actor_dict=actor_dict, mesh_color='beige', texture_name='bone', use_textures=False, rotate=rotate, translate=translate, scale=(s, s, s))
    read_directory(skull_dir, actor_dict=actor_dict, mesh_color='beige', texture_name='bone', use_textures=False, rotate=rotate, translate=translate, scale=(s, s, s))

    actor_dict.update({'skin': body_actor})

    k =     ['skin', 
            'Veins' , 
            'Arteries', 
            'Heart', 
            'Digestive System', 
            'Lymphatic System', 
            'Reproductive System']

    return actor_dict, k
