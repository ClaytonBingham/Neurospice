import numpy as np
import random
from math import sqrt
import pygmsh


class SphericalNodes():
	
	'''
	SpericalNodes class generates a spherical distribution of nodes such that the density of nodes lies along a smooth gradient leading away from the center. Arguments are:
	
	radius = microns
	node_count = the number of nodes user wishes to use to distribute throughout the points_on_sphere
	interval = the space between shells where nodes lie at a given radius between the center and outermost surface of the spherical mesh. (microns)
	
	Usage:
	import neurospice
	from neurospice.BasicGeometries import SphericalNodes
	nodes = SphericalNodes(radius=100,node_count=10000,interval=5)
	
	'''
	
	def __init__(self,radius,node_count,interval):
		nodes = 0
		slope = 0
		if node_count > 100000:
			print('Warning: such a high node count will likely result in a mesh too complex to converge')
		
		while nodes<node_count:
			nodes = sum([int(r*slope) for r in range(1,radius,interval)])
			slope+=0.001
		
		print('Exact number of nodes: '+str(nodes))
		self.xs,self.ys,self.zs = self.get_full_spherical_mesh(range(1,radius,interval),[int(r*slope) for r in range(1,radius,interval)])
	
	
	def points_on_sphere(self,N):
		'''
		This class method takes a number of nodes and distributes them randomly on the surface of a sphere
		
		N = number of nodes
		returns:float,float,float
		'''
		lincr = 2*np.pi / ((1 + np.sqrt(5)) / 2 ) 
		dz = 2.0 / float(N)    
		bands = np.arange(N)       
		z = bands * dz - 1 + (dz/2) 
		r = np.sqrt(1 - z*z)       
		az = random.random()*np.pi+bands * lincr
		x = r * np.cos(az)
		y = r * np.sin(az)
		return(x, y, z)
	
	
	def get_full_spherical_mesh(self,rad_range,n_range):
		'''
		
		This class method takes a range of radii and the number of points found at each radius and returns a set of nodes that can later be meshed.
		rad_range= 1d-array or list of radii in microns
		n_range= 1d-array or list of integer node counts that correspond to each radius in rad_range
		returns: list,list,list
		'''
		
		n_range = list(n_range)
		xs = np.array([])
		ys = np.array([])
		zs = np.array([])
		for n in n_range:
			if n != 0:
				x,y,z = self.points_on_sphere(int(n))
				xs = np.append(xs,x*rad_range[n_range.index(n)])
				ys = np.append(ys,y*rad_range[n_range.index(n)])
				zs = np.append(zs,z*rad_range[n_range.index(n)])
		return(xs,ys,zs)

class UnstructuredNodes():
	
	'''
	
	UnstructuredNodes class takes a maximum internode distance and a maximum number of nodes and returns a idealized distribution of node_count nodes such that Delaunay triangulated meshing of these nodes generates tetrahedral elements that are mostly equifacial
	
	side_lengths = maximum internode distance in microns
	node_count = the number of nodes to be used in the mesh
	shift = the x,y,z location of the bottom/left/back corner of the bounding box of the mesh--a node will be placed here.
	
	Usage:
	import neurospice
	from neurospice.BasicGeometries import UnstructuredNodes
	nodes = UnstructuredNodes([100,100,100], 1001, shift=[0,0,0])
	
	'''
	
	def __init__(self,side_lengths, node_count, shift=[0,0,0]):
		#geom = pygmsh.opencascade.Geometry(characteristic_length_max=40.0)
		geom = pygmsh.opencascade.Geometry()
		rectangle = geom.add_rectangle(shift, side_lengths[0], side_lengths[1])
		geom.extrude(rectangle, [0, 0, side_lengths[2]])
		npoints = 0
		sidelen = min(side_lengths)/2.0
		while npoints <= node_count:
			msh = pygmsh.generate_mesh(geom,extra_gmsh_arguments=['-clmax',str(sidelen),'-v','0'])
			self.points = msh.points
			self.cells = msh.cells
			self.point_data = msh.point_data
			self.cell_data = msh.cell_data
			self.field_data = msh.field_data
			sidelen*=0.99
			npoints = len(self.points)
		
		self.xs = [p[0] for p in self.points]
		self.ys = [p[1] for p in self.points]
		self.zs = [p[2] for p in self.points]

class CubicNodes():
	
	'''
	
	CubicNodes evenly distributes a user-designated number of nodes throughout a user-designated bounding box such that nodes lie on a grid with all internode distances being equal.
	
	side_lengths = [x,y,z] where x,y,and z are float or integer values that represent the absolute lengths of the bounding box dimensions.
	node_count = the maximum number of nodes that can be used to populate the grid
	shift = [x,y,z] where x,y,z are float or integer values that determine the location of the bottom, left, back corner of the grid
	
	Usage:
	import neurospice
	from neurospice.BasicGeometries import CubicNodes
	nodes = CubicNodes([100,100,100], 1001, shift=[0,0,0])
	'''
	
	
	def __init__(self,side_lengths, node_count, shift=[0,0,0]):
		nodes = 0
		cube_side_len = self.get_element_dimensions(node_count,side_lengths)
		print('Element side length: '+str(cube_side_len)+' '+u"\u03bcm")
		print('Exact number of nodes: '+str(int(side_lengths[0]/cube_side_len)*int(side_lengths[1]/cube_side_len)*int(side_lengths[2]/cube_side_len)))
		self.xs,self.ys,self.zs = self.get_full_cubic_mesh(cube_side_len,side_lengths)
		self.xs = [x+shift[0] for x in self.xs]
		self.ys = [y+shift[1] for y in self.ys]
		self.zs = [z+shift[2] for z in self.zs]
		self.sort_xyz_by_bottom_corner()
	
	
	def sort_xyz_by_bottom_corner(self):
		'''
		class method to sort points by distance from bottom back left corner of grid
		'''
		ol = self.get_ordered_list(list(zip(self.xs,self.ys,self.zs)),(self.xs[0],self.ys[0],self.zs[0]))
		self.xs = [item[0] for item in ol]
		self.ys = [item[1] for item in ol]
		self.zs = [item[2] for item in ol]
	
	def get_ordered_list(self,points,b):
		'''
		class method to sort points by distance from user designated point
		points = [[x,y,z],[x1,y1,z1],...] grid locations
		b = [x,y,z] reference location
		returns:list of lists [x,y,z]
		'''
		points.sort(key = lambda p: sqrt((p[0] - b[0])**2 + (p[1] - b[1])**2 + (p[2] - b[2])**2))
		return(points)
	
	
	def get_element_dimensions(self,node_count,side_lengths):
		'''
		class method to calculate element size in grid model
		node_count = integer, user-designated target quantity of elements in mesh
		side_lengths = [float,float,float] corresponding to the dimensions of the volume to be meshed
		returns:float
		'''
		cube_side_len = np.max(side_lengths)
		nc = 0
		while nc < node_count:
			cube_side_len*=0.999
			nc = (int(side_lengths[0]/cube_side_len))*(int(side_lengths[1]/cube_side_len))*(int(side_lengths[2]/cube_side_len))
		
		print((int(side_lengths[0]/cube_side_len)-1)*(int(side_lengths[1]/cube_side_len)-1)*(int(side_lengths[2]/cube_side_len)-1), 'number of cubes')
		return(cube_side_len)
	
	def get_full_cubic_mesh(self,cube_side_len,side_lengths):
		'''
		class method to generate a grid arrangement with uniform distribution of nodes to be meshed
		cube_side_len = float, element length
		side_lengths = [float,float,float] dimensions of the volume to be meshed
		returns:list of floats, list of floats, list of floats
		'''
		xs = []
		ys = []
		zs = []
		plane_dims_x = np.arange(0,side_lengths[0],cube_side_len)
		plane_dims_y = np.arange(0,side_lengths[1],cube_side_len)
		plane_dims_z = np.arange(0,side_lengths[2],cube_side_len)
		for xinc in plane_dims_x:
			for yinc in plane_dims_y:
				for zinc in plane_dims_z:
					xs.append(xinc)
					ys.append(yinc)
					zs.append(zinc)
		
		return(xs,ys,zs)
