import numpy as np
import vtk
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from math import sqrt

class VisualizeTetMesh():
	
	'''
	class that creates and visualizes a tetrahedral mesh using previously defined mesh nodes
	xs,ys,zs=[x,x1,...],[y,y1,...],[z,z1,...] corresponding to locations of nodes to be meshed
	step_alpha=integer or float meaning the maximum connection distance of Delaunay triangulation (defualt=3200)
	plot= boolean indicating whether the mesh is plotted in vtk or not (default=True)
	save= boolean indicating whether the mesh vtk actor is saved as .png or not (default=True)
	
	Usage:
	import neurospice
	from neurospice.MeshVisualizer import VisualizeTetMesh
	#VisualizeTetMesh is meant to be called after CubicNodes or UnstructuredNodes (nodes)
	msh = VisualizeTetMesh(nodes.xs,nodes.ys,nodes.zs,3200) #plot vtk mesh and output image : default args plot=True, save=True
	'''
	
	def __init__(self,xs,ys,zs,step_alpha=3200,plot=True,save=True):
		self.hull = ConvexHull(np.array([np.array([xs[i],ys[i],zs[i]]) for i in range(len(xs))]))
		self.profile = self.feed_points_to_polydatavtk(xs,ys,zs)
		hullpoints = []
		for simp in self.hull.simplices:
			for ind in simp:
				hullpoints.append([xs[ind],ys[ind],zs[ind]])
		
		self.hull_profile = self.feed_points_to_polydatavtk([item[0] for item in hullpoints],[item[1] for item in hullpoints],[item[2] for item in hullpoints])
		self.hull_profile = self.delaunay_triangulate_polydata(self.hull_profile,tol=1,alpha=1)
		self.hull_ob = self.build_vtkactor(self.hull_profile,color=[1,0,0],point_resize=True)
		self.profile = self.feed_points_to_polydatavtk(xs,ys,zs)
		self.delny = self.delaunay_triangulate_polydata(self.profile,tol=1,alpha=step_alpha)
		
		if plot or save:
			self.delny = self.shrink_elements(self.delny,factor=0.9)
			self.triangulation = self.build_vtkactor(self.delny,color = [0.85,0.85,0.85])
			ren,iren,renWin = self.build_vtkrenderer(background = (0,0,0), size = (750,750))
			ren.AddActor(self.triangulation)
			ren.AddActor(self.hull_ob)
			
			
			#cam1 = ren.GetActiveCamera()
			#cam1.Zoom(1.5)
			camera = vtk.vtkCamera()
			camera.SetPosition(1,1,1)
			camera.SetFocalPoint(0,0,0)
			ren.SetActiveCamera(camera)
			ren.ResetCamera()
			
			renWin.Render()
			#w2if = vtk.vtkWindowToImageFilter()
			#w2if.SetInput(renWin)
			#w2if.Update()
			iren.Initialize()
			iren.Start()
			if save:
				w2if = vtk.vtkWindowToImageFilter()
				w2if.SetInput(renWin)
				w2if.Update()
				self.write_vtk_image(w2if,str(step_alpha)+'.png')
	
	def feed_points_to_polydatavtk(self,xs,ys,zs):
		'''
		class method to load a vtk polydata object with a cloud of points
		xs,ys,zs=[x,x1,...],[y,y1,...],[z,z1,...] indicating locations of nodes to be meshed
		return:vtk polydata object
		'''
		math = vtk.vtkMath()
		points = vtk.vtkPoints()
		for i in range(len(xs)):
			points.InsertPoint(i,xs[i],ys[i],zs[i])
		
		profile = vtk.vtkPolyData()
		profile.SetPoints(points)
		return(profile)

	def delaunay_triangulate_polydata(self,polydata,tol=1,alpha=50):
		'''
		class method to triangulate points
		polydata=vtk polydata object loaded with point cloud
		tol=integer indicating mesh quality tolerance
		alpha=integer/float indicating the maximum connection distance in the triangulation
		returns:vtk 3d tetrahedralization
		'''
		delny = vtk.vtkDelaunay3D()
		delny.SetInputData(polydata)
		delny.SetTolerance(tol)
		delny.SetAlpha(alpha)
		delny.BoundingTriangulationOff()
		return(delny)

	def shrink_elements(self,ob,factor=0.9):
		'''
		class method to shrink the elements slightly to enhance visualization esthetics
		ob=vtk 3d tetrahedralization
		factor=float 0 to 1 indicating element scaling
		returns:vtk filtered vtk tetrahedralization
		'''
		shrink = vtk.vtkShrinkFilter()
		shrink.SetInputConnection(ob.GetOutputPort())
		shrink.SetShrinkFactor(0.9)
		return(shrink)

	def build_vtkactor(self,polydata_ob,color = [0.85, 0.85, 0.85],point_resize=False):
		'''
		class method that converts a vtk object into an actor to be set in a vtk scene
		polydata_ob=vtk object
		color=RBG used to paint the elements (default=[0.85,0.85,0.85])
		point_resize=boolean option to resize the points rendered (default=False)
		returns:vtk actor
		'''
		mapper = vtk.vtkDataSetMapper()
		mapper.SetInputConnection(polydata_ob.GetOutputPort())
		actor = vtk.vtkActor()
		actor.SetMapper(mapper)
		actor.GetProperty().SetColor(color[0],color[1],color[2])
		if point_resize:
			actor.GetProperty().SetPointSize(3.0)
		
		return(actor)


	def vtk_ascii_write(self,unstructuredgrid_object,fname):
		'''
		class method that writes a vtk unstructured grid object to a file (save the mesh)
		unstructuredgrid_object=vtk unstructured grid object
		fname=text string with path/filename target for writing mesh
		'''
		pa = vtk.vtkPassArrays()
		pa.SetInputConnection(unstructuredgrid_object.GetOutputPort())
		writer = vtk.vtkUnstructuredGridWriter()
		writer.SetFileName(fname)
		writer.SetInputConnection(pa.GetOutputPort())
		writer.Update()
		writer.Write()


	def vtk_unstructured_grid_to_stl(self,vtkfname,stlfname):
		'''
		class method used to write mesh to .stl format
		vtkfname=filename where vtk mesh is saved
		stlfname=filename where .stl formatted mesh is to be written
		'''
		reader = vtk.vtkUnstructuredGridReader()
		reader.SetFileName(vtkfname)
		surface_filter = vtk.vtkDataSetSurfaceFilter()
		surface_filter.SetInputConnection(reader.GetOutputPort())
		triangle_filter = vtk.vtkTriangleFilter()
		triangle_filter.SetInputConnection(surface_filter.GetOutputPort())
		writer = vtk.vtkSTLWriter()
		writer.SetFileName(stlfname)
		writer.SetInputConnection(triangle_filter.GetOutputPort())
		writer.Write()

	def build_vtkrenderer(self,background = (0,0,0), size = (750,750)):
		'''
		class method that builds a vtk scene to be rendered
		background=RGB colors (default=(0,0,0))
		size=tuple containing the pixel dimensions of the scene to be rendered (default=(750,750))
		returns:vtk renderer,vtk window interactor,vtk renderer window
		'''
		ren = vtk.vtkRenderer()
		renWin = vtk.vtkRenderWindow()
		renWin.AddRenderer(ren)
		iren = vtk.vtkRenderWindowInteractor()
		iren.SetRenderWindow(renWin)
		ren.SetBackground(background[0],background[1],background[2])
		renWin.SetSize(size[0],size[1])
		return(ren,iren,renWin)

	def write_vtk_image(self,vtkimage,fname):
		'''
		class method to save a vtk scene as a .png
		vtkimage=vtk image to window filtered vtk renderer window
		fname=target filename where .png will be saved
		'''
		writer = vtk.vtkPNGWriter()
		writer.SetFileName(fname)
		writer.SetInputData(vtkimage.GetOutput())
		writer.Write()



class VisualizeHexMesh():
	
	'''
	create and visualize a uniform hexahedral mesh using previously defined structured grid-based mesh nodes
	xs,ys,zs=[x,x1,...],[y,y1,...],[z,z1,...] corresponding to locations of nodes to be meshed
	plot= boolean indicating whether the mesh is plotted in vtk or not (default=True)
	save= boolean indicating whether the mesh vtk actor is saved as .png or not (default=True)
	
	Usage:
	import neurospice
	from neurospice.MeshVisualizer import VisualizeHexMesh
	#VisualizeHexMesh is meant to be called after CubicNodes or UnstructuredNodes (nodes)
	msh = VisualizeHexMesh(nodes.xs,nodes.ys,nodes.zs)
	'''
	
	def __init__(self,xs,ys,zs,plot=True,save=True):
		self.xr = (min(xs),max(xs))
		self.yr = (min(ys),max(ys))
		self.zr = (min(zs),max(zs))
		self.vertices = list(zip(xs,ys,zs))
		#self.vertices = self.get_ordered_list(vertices,vertices[0])
		self.simplex_coords = self.create_simplices()
		self.cubes = self.simplices_to_cubes()
		if plot:
			if save:
				self.render(png=True)
			else:
				self.render(png=False)
		
		self.simplices = self.simplex_coords_to_indices()
		
	def get_ordered_list(self,points,b):
		'''
		class method that sorts a list of points by distance from a user designated point
		points=[[x,y,z].[x1,y1,z1],...] locations
		b=[x,y,z] reference point
		returns:list
		'''
		points.sort(key = lambda p: sqrt((p[0] - b[0])**2 + (p[1] - b[1])**2 + (p[2] - b[2])**2))
		return(points)
		points.sort(key = lambda p: sqrt((p[0] - b[0])**2 + (p[1] - b[1])**2 + (p[2] - b[2])**2))
		return(points)
	
	def change_precision(self,lst,precision):
		'''
		class method to change the floating point precision of members of a list
		lst=list of values
		precision=decimal places to preserve in arithmetical operations
		returns:list
		'''
		return([round(item,precision) for item in lst])
	
	def simplex_coords_to_indices(self):
		'''
		class method to lookup coordinate indices from list of nodes for each coordinate in a list of element simplices
		returns:list of lists
		'''
		simplices = []
		vertices_local = [self.change_precision(vert,2) for vert in self.vertices]
		for cube in self.simplex_coords:
			try:
				simplices.append([vertices_local.index(self.change_precision(vert,2)) for vert in cube])
			except:
				print(cube)
		
		return(simplices)
	
	
	def find_cube_side_len(self):
		'''
		class method to find the element size in a grid-based uniform hexahedral mesh
		returns:float
		'''
		distances = cdist([self.vertices[0]],self.vertices[1:])
		return(distances[0][distances.argmin()])
	
	def cube_coords(self,sidelen,xind,yind,zind):
		'''
		class method to create a unit cube and scale it to the element size and shift it to an assigned location within the larger mesh
		sidelen=float
		xind=column index in whole grid
		yind=row index in mesh grid
		zind=stack index in mesh grid
		returns:list of tuples (x,y,z)
		'''
		basiscords = []
		for x in [0,1]:
			for y in [0,1]:
				for z in [0,1]:
					basiscords.append([x,y,z])
		
		basisvoxel = [sidelen*np.array(vert) for vert in basiscords]
		shift = sidelen*np.array([xind,yind,zind])+np.array([self.xr[0],self.yr[0],self.zr[0]])
		return([tuple(shift+item) for item in basisvoxel])
	
	def create_simplices(self):
		'''
		class method to create hex grid-based mesh using predetermined bounding box
		returns:list of lists (element simplices)
		'''
		sidelen = self.find_cube_side_len()
		xsegments = int(round((self.xr[1]-self.xr[0])/sidelen))
		ysegments = int(round((self.yr[1]-self.yr[0])/sidelen))
		zsegments = int(round((self.zr[1]-self.zr[0])/sidelen))
		simplices = []
		print(sidelen,xsegments,ysegments,zsegments,'segments')
		for xind in range(xsegments):
			for yind in range(ysegments):
				for zind in range(zsegments):
					simplices.append(self.cube_coords(sidelen,xind,yind,zind))
		
		return(simplices)
	
	def simplices_to_cubes(self):
		'''
		class method to make an vtk actor from coordinates of a cubic voxel
		returns:vtk actor
		'''
		cubes = []
		for simplex in self.simplex_coords:
			cubes.append(self.makecube(simplex))
		
		return(cubes)
	
	def setup_vtk_objects(self):
		'''
		class method that instantiates vtk objects that are to be filled with points corresponding to nodes in a cubic voxel
		returns:vtk polydata object, vtk points array, vtk cell array, vtk float array
		'''
		cube_ob    = vtk.vtkPolyData()
		points  = vtk.vtkPoints()
		polys   = vtk.vtkCellArray()
		scalars = vtk.vtkFloatArray()
		return(cube_ob,points,polys,scalars)
	
	def shrink_element(self,vertices,factor=0.9):
		'''
		class method to shrink the elements slightly to enhance visualization esthetics
		ob=vtk 3d tetrahedralization
		factor=float 0 to 1 indicating element scaling
		returns:vtk filtered vtk tetrahedralization
		'''
		vertices = np.array([np.array(vert) for vert in vertices])
		cube_centroid = np.average(vertices,axis=0)
		new_vertices = cube_centroid+(vertices-cube_centroid)*factor
		return(new_vertices)
	
	def mkVtkIdList(self,it):
		'''
		class method that Makes a vtkIdList from a Python iterable. 
		returns:vtk id list
		'''
		vil = vtk.vtkIdList()
		for i in it:
			vil.InsertNextId(int(i))
		
		return (vil)
	
	def makecube(self,vertices,colors=None): 	
		'''
		class method that takes the vertices of a cubic voxel and returns a vtk actor
		vertices = array of 8 3-tuples of float representing the vertices of a cube
		colors=RGB to paint elements (default=None)
		returns:vtk actor
		'''
		x = self.shrink_element(vertices,0.9)
		pts = [[0,1,3,2],[0,1,5,4],[0,4,6,2],[7,6,4,5],[7,5,1,3],[7,6,2,3]]
		cube_ob,points,polys,scalars = self.setup_vtk_objects()
		
		# Load the point, cell, and data attributes.
		for i in range(8):
			points.InsertPoint(i, x[i])
		
		for i in range(6):
			polys.InsertNextCell( self.mkVtkIdList(pts[i]) )
		
		if colors != None:
			for i in range(8):
				scalars.InsertTuple1(colors[i],colors[i])
	
		# We now assign the pieces to the vtkPolyData.
		cube_ob.SetPoints(points)
		cube_ob.SetPolys(polys)
		if colors!=None:
			cube_ob.GetPointData().SetScalars(scalars)

		# Now we'll look at it.
		cubeMapper = vtk.vtkPolyDataMapper()
		if vtk.VTK_MAJOR_VERSION <= 5:
			cubeMapper.SetInput(cube_ob)
		
		else:
			cubeMapper.SetInputData(cube_ob)
		
		if colors!=None:
			cubeMapper.SetScalarRange(0,7)
		
		cubeActor = vtk.vtkActor()
		cubeActor.SetMapper(cubeMapper)
		return(cubeActor)
	
	def render(self,png=True,imname='hexmesh.png'):
		'''
		class method that builds a vtk scene to be rendered
		png=boolean to determine whether to save image as .png (default=True)
		imname=filename target for image saving (default='hexmesh.png')
		'''
		# The usual rendering stuff.
		camera = vtk.vtkCamera()
		camera.SetPosition(1,1,1)
		camera.SetFocalPoint(0,0,0)
		renderer = vtk.vtkRenderer()
		renWin   = vtk.vtkRenderWindow()
		renWin.AddRenderer(renderer)
		iren = vtk.vtkRenderWindowInteractor()
		iren.SetRenderWindow(renWin)
		for actor in self.cubes:
			renderer.AddActor(actor)
		
		renderer.SetActiveCamera(camera)
		renderer.ResetCamera()
		renderer.SetBackground(0,0,0)
		renWin.SetSize(300,300)
		
		# interact with data
		renWin.Render()
		iren.Start()
		if png:
			w2if = vtk.vtkWindowToImageFilter()
			w2if.SetInput(renWin)
			w2if.Update()
			self.write_vtk_image(w2if,imname)
	
	def write_vtk_image(self,vtkimage,fname):
		'''
		class method to save a vtk scene as a .png
		vtkimage=vtk image to window filtered vtk renderer window
		fname=target filename where .png will be saved
		'''
		writer = vtk.vtkPNGWriter()
		writer.SetFileName(fname)
		writer.SetInputData(vtkimage.GetOutput())
		writer.Write()


