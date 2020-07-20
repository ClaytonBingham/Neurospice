import pickle
from .ConvenientTools import helpme,timeit
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool 
from mayavi import mlab
import time
#import pytetgen
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
import traceback
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib64'
import PySpice.Logging.Logging as Logging
import logging
logger = Logging.setup_logging()
logger.setLevel(logging.ERROR)
from PySpice.Plot.BodeDiagram import bode_diagram
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import PySpice
PySpice.Spice.Simulation.CircuitSimulator.DEFAULT_SIMULATOR = 'ngspice-subprocess'
#PySpice.Spice.Simulation.CircuitSimulator.DEFAULT_SIMULATOR = 'ngspice-shared'
from scipy.spatial.distance import cdist
from math import sqrt
import mpmath as trig


class AM_circuit():
	
	'''
	translate a netlist into ngSpice elements
	'''
	
	def __init__(self):
		pass
	
	def load_netlist(self,netlistfile):
		'''
		class method to build a circuit model from a netlist
		netlistfile=path/filename that contains a valid spice netlist
		'''
		with open(netlistfile,'r',encoding='utf-8') as f:
			self.netlist = [item.strip() for item in f.readlines()[:-1]]
			self.circuitname = self.netlist[0].lstrip('.title').strip()
			self.instantiate_circuit()
	
	def instantiate_circuit(self):
		'''
		class method that instantiates a circuit model from an inherited netlist
		'''
		self.circuit = Circuit(self.circuitname)
		for line in self.netlist[1:]:
			try:
				self.add_circuit_element_from_netlist_line(line)
			except:
				print(traceback.format_exc())
		
	def add_circuit_element_from_netlist_line(self,line):
		'''
		class method to parse a line from a netlist and add RC components to a circuit model
		line=text string from netlist
		'''
		line = line.split(' ')
		if 'R' in line[0]:
			self.circuit.Resistor(line[0].lstrip('R'),line[1],line[2],kilo(float(line[3].rstrip('k'))))
		
		if 'C' in line[0]:
			self.circuit.Capacitor(line[0].lstrip('C'),line[1],line[2],float(line[3])@u_uF)
		
		if 'I' in line[0]:
			self.circuit.CurrentSource(line[0].lstrip('I'),line[1],line[2],line[3])
	
	def get_nodal_voltages(self,analysis):
		'''
		class method to retrieve nodal voltages from a circuit analysis
		analysis:circuit analysis (simulator.operating_point(), etc.)
		returns:list
		'''
		nvs = {}
		for key in analysis.nodes.keys():
			nvs[int(key)-1] = analysis[key].T.tolist()[0]
		
		return(nvs)

class BuildHexCircuit():
	
	'''
	take all nodes and current sources and create a valid netlist which can be solved by a SPICE solver.
	
	xs,ys,zs = [x1,x2,...],[y1,y2,...],[z1,z2,...] points of the mesh as constructed by CubicNodes or UnstructuredNodes (ordered)
	outputname = user-designated name for netlist generated within this class (default='')
	tsteps = the range of timesteps in the series of solutions e.g. [1,2,3,4,...] as they were originally provided by the user to BuildSolution class in SolutionBuilder.py
	sources = class object SplitCurrents() or ShiftCurrents() (default={} i.e. no current sources at all)
	
	Usage:
	import neurospice
	from neurospice.SolutionBuilder import *
	
	'''
	
	def __init__(self,xs,ys,zs,outputname='',tstep=None,sources={}):
		if tstep ==None:
			print('tstep must be a time-series in a 2d-array. If only a single circuit needs be solved follow this pattern tstep = [[0]]')
			return()
		
		self.tstep = tstep
		self.sources = sources
		self.outputname = outputname
		self.xr = (min(xs),max(xs))
		self.yr = (min(ys),max(ys))
		self.zr = (min(zs),max(zs))
		vertices = list(zip(xs,ys,zs))
		self.vertices = self.get_ordered_list(vertices,vertices[0])
		self.simplex_coords = self.create_simplices()
		self.simplices = self.simplex_coords_to_indices()
		self.write_to_netlist(self.simplices,self.get_hull_nodes(),[item[0] for item in self.vertices],[item[1] for item in self.vertices],[item[2] for item in self.vertices])
	
	def get_hull_nodes(self):
		
		'''
		
		class method that finds the hull nodes of the mesh
		returns:boundary points of mesh [[x,y,z],[x1,y1,z1],...]
		'''
		
		eight_corners = [
			(self.xr[0],self.yr[0],self.zr[0]),
			(self.xr[0],self.yr[0],self.zr[1]),
			(self.xr[0],self.yr[1],self.zr[0]),
			(self.xr[0],self.yr[1],self.zr[1]),
			(self.xr[1],self.yr[0],self.zr[0]),
			(self.xr[1],self.yr[0],self.zr[1]),
			(self.xr[1],self.yr[1],self.zr[0]),
			(self.xr[1],self.yr[1],self.zr[1])
			]
		
		corners = [self.vertices.index(corner) for corner in eight_corners]
		return(corners)
	
	def get_ordered_list(self,points,b):
		
		'''
		
		class method that orders the list by distance from the a given point
		
		points = [[x1,x2,...],[y1,y2,...],[z1,z2,...]] each value is float or integer
		
		b = [x,y,z] where each value is a float or integer. The sort is performed by the distance from b
		
		returns: list of lists of floats sorted by distance from b
		
		'''
		
		
		points.sort(key = lambda p: sqrt((p[0] - b[0])**2 + (p[1] - b[1])**2 + (p[2] - b[2])**2))
		return(points)
	
	
	def change_precision(self,lst,precision):
		
		'''
		
		changes the floating point precision of values in a list
		returns: list of floats
		
		'''
		
		return([round(item,precision) for item in lst])
	
	def simplex_coords_to_indices(self):
		
		
		'''
		
		class method to convert the locations of simplices of mesh elements into indices of simplices from the original node list provided to class
		returns: list of lists of integers
		
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
		
		class method to find the element edge length in a grid-like uniform mesh
		
		'''
		
		
		distances = cdist([self.vertices[0]],self.vertices[1:])
		return(distances[0][distances.argmin()])
	
	def cube_coords(self,sidelen,xind,yind,zind):
		
		'''
		
		class method to generate the reference coordinates of a cubic voxel based on the element side-length and the cardinal order of the voxel in the mesh
		returns: list of tuples of floats (x,y,z)
		
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
		
		class method that voxelizes a bounding box with uniform cubic grid of nodes
		returns: list of lists of tuples of floats (x,y,z)
		
		'''
		
		
		sidelen = self.find_cube_side_len()
		xsegments = int(round((self.xr[1]-self.xr[0])/sidelen))
		ysegments = int(round((self.yr[1]-self.yr[0])/sidelen))
		zsegments = int(round((self.zr[1]-self.zr[0])/sidelen))
		simplices = []
		for xind in range(xsegments):
			for yind in range(ysegments):
				for zind in range(zsegments):
					simplices.append(self.cube_coords(sidelen,xind,yind,zind))
		
		return(simplices)
	
	def  distance_3d(self,target,source):
		
		'''
		
		target=[x,y,z] where each is a float or integer representing a point in space
		source=[x,y,z] where each is a float or integer representing a point in space
		returns:float, euclidean distance between target and source
		
		'''
		
		return(((target[0]-source[0])**2+(target[1]-source[1])**2+(target[2]-source[2])**2)**0.5)
	
	
	def return_nodepairs_from_simplices(self,simps):
		
		'''
		
		class method that reorders simplices to represent edges in a resistor mesh
		simps=list of lists of lists of floats corresponding to mesh simplices
		returns: list of lists of lists of integers
		
		'''
		Rs = {}
		nodepairs = []
		for simp in simps:
			for pair in [[0,2],[0,1],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]]:
				if [simp[pair[0]],simp[pair[1]]] not in nodepairs:
					nodepairs.append((simp[pair[0]],simp[pair[1]]))
					Rs[nodepairs[-1]] = []
		
		return(nodepairs,Rs)
	
	def convert_resistivity_to_resistance(self,A,B,resistivity,crossarea):
		
		'''
		
		class method that converts internode distances into edge resistances by referencing the local resistivity
		
		A = list of floats,integers [x,y,z] corresponding to a mesh node locations
		B = list of floats,integers [x,y,z] corresponding to a mesh node locations
		resistivity=local resistivity
		crossarea = crossectional area of the wire connecting A and B
		
		returns: float corresponding to the connection resistance
		
		'''
		
		
		dist = self.distance_3d(A,B)
		return(resistivity*float(dist)/crossarea)


	def resolve_joint_conductances(self,Rs):		
		'''
		class method that sums parallel resistors to re-unify the mesh and have a single value for every node pair
		'''
		for pair in Rs.keys():
			Rs[pair] = 1.0/(np.sum([1.0/resistance for resistance in Rs[pair]]))
		
		return(Rs)
		
	def return_resistor_strings_from_nodepairs(self,Rs,nodepairs,xs,ys,zs):
		
		'''
		
		class method to consider all node pairs in the mesh and convert them into spice netlist format with a valid resistance
		nodepairs=list of list of integers corresponding to the node indices of mesh connections
		xs,ys,zs = [x1,x2,...],[y1,y2,...],[z1,z2,...] points of the mesh as constructed by CubicNodes or UnstructuredNodes (ordered)
		returns:text string which forms the block of text containing all resistors to be written to the .netlist
		
		'''
		
		
		rblock = ''
		for nodepair in nodepairs:
			Rs[nodepair].append(self.convert_resistivity_to_resistance([xs[nodepair[0]],ys[nodepair[0]],zs[nodepair[0]]],[xs[nodepair[1]],ys[nodepair[1]],zs[nodepair[1]]],3.0,0.25))
		
		Rs = self.resolve_joint_conductances(Rs)
		
		
		for n,nodepair in enumerate(Rs.keys()):
			rblock = rblock+'R'+str(n+1)+' '+str(nodepair[0]+1)+' '+str(nodepair[1]+1)+' '+str(Rs[nodepair])+'k\n'
		
		return(rblock,n)

	def add_ground_to_periphery_and_rblock(self,Rindex,rblock,peripheral_nodes):
		
		'''
		
		class method to connect mesh boundary nodes to the ground
		
		Rindex=number of resistors in rblock
		rblock=text string which forms the block of text containing all resistors to be written to the .netlist
		peripheral_nodes=list of node ids for nodes which lie at the boundary of the mesh
		returns:text string which forms the block of text containing all resistors to be written to the .netlist
		
		'''
		
		
		for n,node in enumerate(peripheral_nodes):
			rblock = rblock+'R'+str(n+2+Rindex)+' 0 '+str(node+1)+' 0.00000000000001k\n'
		
		return(rblock)
	
	def write_sources_to_footer(self):
		
		'''
		
		class method that writes the current sources to the netlist
		returns:text string containing the block of text containing all current sources to be written to the .netlist
		
		'''
		
		footer = ''
		source_count = 0
		for key in self.sources.keys():
			if self.sources[key][self.tstep] == 0.0:
				continue
			
			footer = footer+'I'+str(source_count)+' 0 '+str(key+1)+' '+str(self.sources[key][self.tstep])+'uA\n'
			source_count+=1
		
		return(footer+'.end')
	
	def write_to_netlist(self,simplices,peripheral_nodes,xs,ys,zs):
		
		'''
		
		class method that takes a mesh, adds ground and current sources and writes a valid netlist
		
		simplices=list of lists of integers corresponding to the nodes of the circuit
		peripheral_nodes=list if integers corresponding to the ndoes of the circuit found at the boundary of the mesh
		xs,ys,zs = [x1,x2,...],[y1,y2,...],[z1,z2,...] points of the mesh as constructed by CubicNodes or UnstructuredNodes (ordered)
		
		
		
		'''
		
		header = '.title RC Mesh\n'
		if self.sources == {}:
			footer = 'I1 0 1 500uA\n.end'
		
		else:
			footer = self.write_sources_to_footer()
		
		with open(self.outputname,'w') as f:
			nodepairs,Rs = self.return_nodepairs_from_simplices(simplices)
			self.info = str(len(Rs.keys()))+' number of resistors in hexahedral mesh with a node count of: '+str(len(self.vertices))
			rblock,Rindex = self.return_resistor_strings_from_nodepairs(Rs,nodepairs,xs,ys,zs)
			rblock = self.add_ground_to_periphery_and_rblock(Rindex,rblock,peripheral_nodes)
			f.write(header)
			f.write(rblock)
			f.write(footer)

class BuildTetCircuit():
	
	'''
	take all nodes and current sources and create a valid netlist which can be solved by a SPICE solver.
	
	xs,ys,zs = [x1,x2,...],[y1,y2,...],[z1,z2,...] points of the mesh as constructed by CubicNodes or UnstructuredNodes (ordered)
	outputname = user-designated name for netlist generated within this class (default='')
	tsteps = the range of timesteps in the series of solutions e.g. [1,2,3,4,...] as they were originally provided by the user to BuildSolution class in SolutionBuilder.py
	sources = class object SplitCurrents() or ShiftCurrents() (default={} i.e. no current sources at all)
	
	Usage:
	import neurospice
	from neurospice.SolutionBuilder import *
	
	
	'''
	
	def __init__(self,xs,ys,zs,outputname,tstep,sources={},geo='tet',cotan_rule=True):
		self.vertices = list(zip(xs,ys,zs))
		self.tstep = tstep
		self.cotan_rule = cotan_rule
		self.sources = sources
		self.outputname = outputname
		self.msh = Delaunay(np.array([np.array(vertice) for vertice in self.vertices]))
		print('Number of elements: ',len(self.msh.simplices))
		self.write_to_netlist(self.msh.simplices,self.get_hull_nodes())
	
	def  distance_3d(self,target,source):
		'''
		
		target=[x,y,z] where each is a float or integer representing a point in space
		source=[x,y,z] where each is a float or integer representing a point in space
		returns:float, euclidean distance between target and source
		
		'''
		return(((target[0]-source[0])**2+(target[1]-source[1])**2+(target[2]-source[2])**2)**0.5)
	
	
	def get_hull_nodes(self):
		
		'''
		
		class method that finds the hull nodes of the mesh
		returns:boundary points of mesh [[x,y,z],[x1,y1,z1],...]
		
		'''
		
		hull = ConvexHull(np.array([np.array(vertice) for vertice in self.vertices]))
		return(hull.vertices)
	
	
	def resolve_joint_conductances(self):
		'''
		class method that sums parallel resistors to re-unify the mesh and have a single value for every node pair
		'''
		for pair in self.nodepairs.keys():
			self.nodepairs[pair] = 1.0/(np.sum([1.0/resistance for resistance in self.nodepairs[pair]]))
	
	
	
	def find_all_conductances(self):
		'''
		class method that loops through all elements and returns the pairwise conductances*lengths for every node pairing in the tetrahedral mesh
		'''
		self.nodepairs = dict()
		for simp in self.msh.simplices:
			for pair in list(itertools.combinations(simp,2)):
				self.nodepairs[pair] = []
			
			self.assign_conductances_cotan3d(simp)
		
		
		self.resolve_joint_conductances()
	
	
	def assign_conductances_cotan3d(self,simp):
		'''
		class method that takes the simplices of a tetrahedral element and calculates the pairwise cotan_rule conductances of each edge
		'''
		#pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
		pairs = [pair for pair in list(itertools.combinations(simp,2))]
		side_lens = [self.distance_3d(self.vertices[pair[0]],self.vertices[pair[1]]) for pair in pairs]
		verts_from_perms = [
							(0, 1, 2, 3),
							(0, 2, 1, 3),
							(0, 3, 1, 2),
							(1, 2, 3, 0),
							(1, 3, 0, 2),
							(2, 3, 0, 1)]
		
		dihedral_angles = [self.calculate_dihedral_from_vertices([self.vertices[aind[0]],self.vertices[aind[1]],self.vertices[aind[2]],self.vertices[aind[3]]]) for aind in verts_from_perms]
		for p,pair in enumerate(pairs):
			self.nodepairs[pair].append(self.cotan_3d(dihedral_angles[-p],side_lens[-p],3.0))
	
	
	def shift_rays(self,u,v,w):
		minimum = 0
		maximum = 0
		for ray in [u,v,w]:
			if np.min(ray) < minimum:
				minimum = np.min(ray)
			if np.max(ray) > minimum:
				maximum = np.max(ray)
		
		shift = np.abs(maximum - 2*minimum)
		return(u+shift,v+shift,w+shift)
	
	def calculate_dihedral_from_vertices(self,vertices):
		'''
		class method that returns the dihedral angle formed by three co-embarking vectors whose 
		vertices are a,b, c, and d. ab is the edge over which the dehedral of interest is formed.
		'''
		a = np.array(vertices[0])
		b = np.array(vertices[1])
		c = np.array(vertices[2])
		d = np.array(vertices[3])
		u = b-a
		v = c-a
		w = d-a
		u,v,w = self.shift_rays(u,v,w)
		return(float(trig.acot(((np.dot(u,u)**0.5)*np.dot(v,w)-((np.dot(u,v)*np.dot(u,w))/(np.dot(u,u)**0.5)))/(np.dot(u,np.cross(v,w))))))
	
	
	def cotan_3d(self,theta,opposite_side_len,resistivity):
		'''
		
		class method that finds edge resistances by referencing the local resistivity
		resistivity=local resistivity
		opposite_side_len=length of tet edge which shares no nodes with the edge being considered
		theta= dihedral angle formed by two surfaces across from the edge being considered
		returns: float corresponding to the connection resistance
		
		'''
		cotan_3d = (6*resistivity*np.tan(theta))/opposite_side_len
		if cotan_3d < 0.0:
			print('cotan_3d',cotan_3d,theta,opposite_side_len)
		
		return(cotan_3d)
	
	
	
	def return_nodepairs_from_simplices(self,simps):
		'''
		
		class method that reorders simplices to represent edges in a resistor mesh
		simps=list of lists of lists of floats corresponding to mesh simplices
		returns: list of lists of lists of integers
		
		'''
		pairwise_conductances = []
		nodepairs = []
		for simp in simps:
			for pair in list(itertools.combinations(simp,2)):
				nodepairs.append(pair)
		
		return(list(set(nodepairs)))
	
	
	
	def convert_resistivity_to_resistance(self,A,B,resistivity):
		'''
		
		class method that converts internode distances into edge resistances by referencing the local resistivity
		
		A = list of floats,integers [x,y,z] corresponding to a mesh node locations
		B = list of floats,integers [x,y,z] corresponding to a mesh node locations
		resistivity=local resistivity
		crossarea = crossectional area of the wire connecting A and B
		
		returns: float corresponding to the connection resistance
		
		'''
		dist = self.distance_3d(A,B)
		return(resistivity*float(dist)/crossarea)
	
	
	def return_resistor_strings_from_nodepairs(self):
		'''
		
		class method to consider all node pairs in the mesh and convert them into spice netlist format with a valid resistance
		nodepairs=list of list of integers corresponding to the node indices of mesh connections
		xs,ys,zs = [x1,x2,...],[y1,y2,...],[z1,z2,...] points of the mesh as constructed by CubicNodes or UnstructuredNodes (ordered)
		returns:text string which forms the block of text containing all resistors to be written to the .netlist
		
		'''
		
		rblock = ''
		if not self.cotan_rule:
			for n,nodepair in enumerate(self.nodepairs):
				resistance = self.convert_resistivity_to_resistance([self.xs[nodepair[0]],self.ys[nodepair[0]],self.zs[nodepair[0]]],[self.xs[nodepair[1]],self.ys[nodepair[1]],self.zs[nodepair[1]]],2.8,1.0*12**-6)
				rblock = rblock+'R'+str(n+1)+' '+str(nodepair[0]+1)+' '+str(nodepair[1]+1)+' '+str(resistance)+'k\n'
		
		else:
			self.find_all_conductances()
			for n,nodepair in enumerate(list(self.nodepairs.keys())):
				rblock = rblock+'R'+str(n+1)+' '+str(nodepair[0]+1)+' '+str(nodepair[1]+1)+' '+str(self.nodepairs[nodepair])+'k\n'
		
		return(rblock,n)
	
	def add_ground_to_periphery_and_rblock(self,Rindex,rblock,peripheral_nodes):
		'''
		
		class method to connect mesh boundary nodes to the ground
		
		Rindex=number of resistors in rblock
		rblock=text string which forms the block of text containing all resistors to be written to the .netlist
		peripheral_nodes=list of node ids for nodes which lie at the boundary of the mesh
		returns:text string which forms the block of text containing all resistors to be written to the .netlist
		
		'''
		for n,node in enumerate(peripheral_nodes):
			rblock = rblock+'R'+str(n+2+Rindex)+' 0 '+str(node+1)+' 0.00000000000001k\n'
		
		return(rblock)
	
	def write_sources_to_footer(self):
		'''
		
		class method that writes the current sources to the netlist
		returns:text string containing the block of text containing all current sources to be written to the .netlist
		
		'''
		footer = ''
		source_count = 0
		for key in self.sources.keys():
			if self.sources[key][self.tstep] == 0.0:
				continue
			
			footer = footer+'I'+str(source_count)+' 0 '+str(key+1)+' '+str(self.sources[key][self.tstep])+'uA\n'
			source_count+=1
		
		return(footer+'.end')
	
	def write_to_netlist(self,simplices,peripheral_nodes):
		'''
		
		class method that takes a mesh, adds ground and current sources and writes a valid netlist
		
		simplices=list of lists of integers corresponding to the nodes of the circuit
		peripheral_nodes=list if integers corresponding to the ndoes of the circuit found at the boundary of the mesh
		xs,ys,zs = [x1,x2,...],[y1,y2,...],[z1,z2,...] points of the mesh as constructed by CubicNodes or UnstructuredNodes (ordered)
		
		
		
		'''
		header = '.title RC Mesh\n'
		if self.sources == {}:
			footer = 'I1 0 1 500uA\n.end'
		
		else:
			footer = self.write_sources_to_footer()
		
		with open(self.outputname,'w') as f:
			if not self.cotan_rule:
				self.nodepairs = self.return_nodepairs_from_simplices(simplices)
			
			
			rblock,Rindex = self.return_resistor_strings_from_nodepairs()
			self.info = str(len(list(self.nodepairs.keys())))+' number of resistors in tetrahedral mesh with a node count of: '+str(len(self.vertices))
			rblock = self.add_ground_to_periphery_and_rblock(Rindex,rblock,peripheral_nodes)
			f.write(header)
			f.write(rblock)
			f.write(footer)
			del(self.nodepairs)




class SolveCircuit():
	
	'''
	Use ngSpice to solve either an operating point analysis or transient analysis for a given netlist
	
	nodes = [[x1,x2,...],[y1,y2,...],[z1,z2,...]] points of the mesh as constructed by CubicNodes or UnstructuredNodes (ordered)
	netlist=filename of the netlist to be solved
	time_step=the lenth of time to be simulated in a transient analysis (default=0 i.e. defaults to operating point analysis)
	plot=creates an mlab scene based on the nodal voltages solved (default=True)
	
	Usage:
	import neurospice
	from neurospice.SolutionBuilder import *
	#meant to be called after CubicNodes/UnstructuredNodes (nodes), then SplitCurrents/ShiftCurrents (sources)
	solver = BuildSolution(nodes,sources,range(100),geo='hexahedral')
	
	'''
	
	def __init__(self,nodes,netlist,time_step=0,plot=True):
		AM = AM_circuit()
		AM.load_netlist(netlist)
		simulator = AM.circuit.simulator(temperature=37, nominal_temperature=37)
		if time_step != 0:
			self.nvs = AM.get_nodal_voltages(simulator.transient(step_time=time_step@u_us,end_time=time_step@u_us))
		else:
			self.nvs = AM.get_nodal_voltages(simulator.operating_point())
		
		if plot:
			self.plot_nodal_voltages(nodes.xs,nodes.ys,nodes.zs)
	
	
	def  distance_3d(self,target,source):
		'''
		
		target=[x,y,z] where each is a float or integer representing a point in space
		source=[x,y,z] where each is a float or integer representing a point in space
		returns:float, euclidean distance between target and source
		
		'''
		return(((target[0]-source[0])**2+(target[1]-source[1])**2+(target[2]-source[2])**2)**0.5)
	
	
	def plot_nodal_voltages(self,xs,ys,zs):
		'''
		class method plots nodal voltages in an mayavi/mlab scene
		
		xs,ys,zs = [x1,x2,...],[y1,y2,...],[z1,z2,...] points of the mesh as constructed by CubicNodes or UnstructuredNodes (ordered)
		
		
		'''
		x = []
		y = []
		z = []
		nv = []
		for key in self.nvs.keys():
			if self.distance_3d([xs[key],ys[key],zs[key]],[0,0,0]) < 10:
				pass
			else:
				x.append(xs[key])
				y.append(ys[key])
				z.append(zs[key])
				nv.append(self.nvs[key])
		
		mlab.points3d(x,y,z,nv)
		mlab.show()


class BuildSolution():
	
	'''
	
	Class that multithreads the solution of a series of meshes
	nodes = [[x1,x2,...],[y1,y2,...],[z1,z2,...]] points of the mesh as constructed by CubicNodes or UnstructuredNodes (ordered)
	sources = class object SplitCurrents() or ShiftCurrents() (default={} i.e. no current sources at all)
	tsteps = the range of timesteps in the series of solutions e.g. [1,2,3,4,...] as they were originally provided by the user to BuildSolution class in SolutionBuilder.py
	numthreads=integer, the number of threads you wish to use that should probably correspond to the number of cpu's you have or less depending on how much RAM your device has and the size of the current source files you are using.
	geo=the geometry of the mesh to be used (default='tetrahedral')
	'''
	
	def __init__(self,nodes,sources,tsteps,numthreads=multiprocessing.cpu_count(),geo='tetrahedral'):
		if geo not in ['tetrahedral','hexahedral']:
			print('geo must be either "tetrahedral" or "hexahedral"')
			return()
		
		self.nodes = nodes
		self.numthreads=numthreads
		self.sources = sources
		self.pool = ThreadPool(self.numthreads)
		if geo == 'hexahedral':
			self.pool.map(self.build_and_solve_hex,tsteps)
		
		if geo == 'tetrahedral':
			self.pool.map(self.build_and_solve_tet,tsteps)
	
	def build_and_solve_hex(self,tstep):
		
		'''
		class method that builds and solves a hexahedral circuit
		tstep = one value in the range of timesteps in the series of solutions e.g. [1,2,3,4,...] as they were originally provided by the user to BuildSolution class in SolutionBuilder.py
		
		
		'''
		
		timer = timeit('solved timestep in:')
		if self.sources != {}:
			circuit = BuildHexCircuit(self.nodes.xs,self.nodes.ys,self.nodes.zs,str(tstep)+'_netlist.net',tstep,self.sources.multi_pole_sources)
		else:
			circuit = BuildHexCircuit(self.nodes.xs,self.nodes.ys,self.nodes.zs,str(tstep)+'_netlist.net',tstep,self.sources)

		self.info = circuit.info
		solution = SolveCircuit(self.nodes,str(tstep)+'_netlist.net',plot=False) #arg time_step=us for transient response
		with open(str(tstep)+'_solution','wb') as f:
			pickle.dump(solution,f)
		
		timer.timeout()
		
	def build_and_solve_tet(self,tstep):
		
		'''
		class method that builds and solves a tetrahedral circuit
		tstep = one value in the range of timesteps in the series of solutions e.g. [1,2,3,4,...] as they were originally provided by the user to BuildSolution class in SolutionBuilder.py
		
		
		'''
		
		timer = timeit('solved timestep in:')
		if self.sources != {}:
			circuit = BuildTetCircuit(self.nodes.xs,self.nodes.ys,self.nodes.zs,str(tstep)+'_netlist.net',tstep,self.sources.multi_pole_sources)
		else:
			circuit = BuildTetCircuit(self.nodes.xs,self.nodes.ys,self.nodes.zs,str(tstep)+'_netlist.net',tstep,self.sources)
		
		self.info = circuit.info
		solution = SolveCircuit(self.nodes,str(tstep)+'_netlist.net',plot=False) #arg time_step=us for transient response
		with open(str(tstep)+'_solution','wb') as f:
			pickle.dump(solution,f)
		
		timer.timeout()
