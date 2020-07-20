import os
from scipy import spatial
import numpy as np
import pickle
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool 
from scipy.spatial.distance import cdist
from .ConvenientTools import helpme,timeit

class ShiftCurrents():
	
	'''
	Takes model-based current sources and AM-circuit nodelist. Sorts sources according to the nearest node and sums them (i.e. multi-pole source).
	nodeslist = the [[x1,x2,...],[y1,y2,...],[z1,z2,...]] points of the mesh as constructed by CubicNodes or UnstructuredNodes (ordered)
	datadir = text string directory where currents are found
	num_dfiles = integer quantity of files containing current sources
	numthreads = integer number of threads to multithread (default=multiprocessing.cpu_count())
	
	Usage:
	import neurospice, os
	from neurospice.CurrentSourceManager import ShiftCurrents
	datadir=os.getcwd()
	threads=4
	
	#ShuftCurrents is to be instantiated after CubicNodes or Unstructured nodes (nodes)
	sources = ShiftCurrents(list(zip(nodes.xs,nodes.ys,nodes.zs)),datadir,100,numthreads=threads) #(AM-circuit nodes, cellInfo directory)
	
	'''
	
	def __init__(self,nodelist,datadir,num_dfiles,numthreads=multiprocessing.cpu_count()):
		print('Calculating multi-pole current sources from larger set of currents')
		self.numthreads = numthreads
		self.dfile_count = num_dfiles #100 currents + 100 locations = 100 dfiles -- counterintuitive?
		self.nl = np.array(nodelist).reshape(len(nodelist),3)
		self.datadir = datadir+'/'
		self.instantiate_multi_pole_dict()
		self.thread_dfiles()
	
	def reshape_locations(self,locations):
		'''
		class method to ensure locations are shaped as expected
		locations=locations list [[x1,y1,z1],[x2,y2,z2],...]
		returns: d3 numpy array
		'''
		return(np.array(locations).reshape(int(np.array(locations).size/3.0),3))
	
	def build_curr_node_inds(self,locs):
		'''
		class method to find nearest mesh node to locations in locations list
		returns: list of node indices
		'''
		return([row.argmin() for row in cdist(self.reshape_locations(locs),self.nl)])
	
	def instantiate_multi_pole_dict(self):
		'''
		class method that builds an empty dictionary to contain currents corresponding to each circuit node
		returns:dictionary
		'''
		self.multi_pole_sources = dict(zip(range(len(self.nl)),[np.zeros(401) for nd in self.nl]))
	
	def sort_cell_curr_to_poles_v1(self,currents,curr_node_inds):
		'''
		class method that looks up the current amplitude of a given location and adds it to the nodal current to which it was assigned
		currents={}
		curr_node_inds=list of node indices indicating the node to which a given current source was assigned
		'''
		nodes_to_loop = np.array(list(set(curr_node_inds)))
		for nd in nodes_to_loop:
			self.multi_pole_sources[nd]=self.multi_pole_sources[nd]+sum([np.array(currents[cnind]) for cnind in np.where(np.array(curr_node_inds)==nd)[0]])
	
	def sort_cell_curr_to_poles_v2(self,currents,curr_node_inds):
		'''
		class method that looks up the current amplitude of a given location and adds it to the nodal current to which it was assigned
		currents={}
		curr_node_inds=list of node indices indicating the node to which a given current source was assigned
		'''
		for c,curr in currents:
			self.multi_pole_sources[curr_node_inds[c]]=self.multi_pole_sources[curr_node_inds[c]]+np.array(curr)
	
	def handle_cell(self,currents,locations):
		'''
		class method that takes the locations and currents for a single cell and assigns them to the mesh node to which they are nearest
		currents = {}
		locations = [[x,y,z],[x1,y1,z1],...] locations of current sources
		'''
		curr_node_inds = self.build_curr_node_inds(locations)
		self.sort_cell_curr_to_poles_v1(currents,curr_node_inds)
	
	def handle_dfile(self,findex):
		'''
		class method to load currents and locations and assign cell current sources to mesh nodes
		findex=integer identifying an file index in a list of data files
		'''
		with open(self.datadir+str(findex)+'_locations_single','rb') as f:
			locations = pickle.load(f,encoding='latin1')
		with open(self.datadir+str(findex)+'_currents','rb') as f:
			currents = pickle.load(f,encoding='latin1')
		
		for key in locations.keys():
			self.handle_cell(currents[key],locations[key])
	
	def thread_dfiles(self):
		'''
		class method that multithreads the processing of current and location files ultimately resulting in assigning of current sources to mesh nodes
		'''
		timer = timeit('threaded all files in:')
		pool = ThreadPool(self.numthreads)
		pool.map(self.handle_dfile,range(self.dfile_count))
		timer.timeout()


class SplitCurrents():
	
	'''
	Takes model-based current sources and AM-circuit nodelist. Sorts sources according to the simplex/element in which it is found and splits them by distance into each respective node (i.e. multi-pole source).
	nodeslist = the [[x1,x2,...],[y1,y2,...],[z1,z2,...]] points of the mesh as constructed by CubicNodes or UnstructuredNodes (ordered)
	datadir = text string directory where currents are found
	num_dfiles = integer quantity of files containing current sources
	geo=text string indicating the mesh geometry (default='tet'). If not 'tet' then class defaults to 'hex'.
	numthreads = integer number of threads to multithread (default=multiprocessing.cpu_count())
	

	###NEEDS TESTING###
		potential issues:
			1. nearest four/eight nodes not guaranteed to contain point if not structured grid-based mesh
	
	Usage:
	import neurospice, os
	from neurospice.CurrentSourceManager import SplitCurrents
	datadir=os.getcwd()
	
	#SplitCurrents is to be instantiated after CubicNodes or Unstructured nodes (nodes)
	sources = SplitCurrents(list(zip(nodes.xs,nodes.ys,nodes.zs)),datadir,100,geo='hex',numthreads=threads)
	
	'''
	
	def __init__(self,nodelist,datadir,num_dfiles,geo='tet',numthreads=multiprocessing.cpu_count()):
		#scipy.spatial.Delaunay.find_simplex(self, xi, bruteforce=False, tol=None) #critical function for finding simplex containing an arbitrary point
		self.geo=geo
		self.numthreads = numthreads
		print('Calculating multi-pole current sources from larger set of currents')
		self.dfile_count = num_dfiles #100 currents + 100 locations = 100 dfiles -- counterintuitive?
		self.nl = np.array(nodelist).reshape(len(nodelist),3)
		self.datadir = datadir+'/'
		self.instantiate_multi_pole_dict()
		self.thread_dfiles()
	
	
	def reshape_locations(self,locations):
		'''
		class method to ensure locations are shaped as expected
		locations=locations list [[x1,y1,z1],[x2,y2,z2],...]
		returns: d3 numpy array
		'''
		return(np.array(locations).reshape(int(np.array(locations).size/3.0),3))
	
	
	def build_curr_node_inds_tet(self,locs):
		'''
		class method to find nearest tet mesh node to locations in locations list
		locs=[[x,y,z],[x1,y1,z1],...] locations of current sources
		returns: list of node indices, list of current-node distances
		'''
		dists = cdist(self.reshape_locations(locs),self.nl)
		curr_node_inds = []
		curr_node_dists = []
		for r,row in enumerate(dists):
			sortedargs = np.argsort(row)[:4]
			curr_node_inds.append(sortedargs)
			curr_node_dists.append([row[arg] for arg in sortedargs])
		
		
		return(curr_node_inds,curr_node_dists)	
	
	def build_curr_node_inds_hex(self,locs):
		'''
		class method to find nearest hex mesh node to locations in locations list
		locs=[[x,y,z],[x1,y1,z1],...] locations of current sources
		returns: list of node indices, list of current-node distances
		'''
		dists = cdist(self.reshape_locations(locs),self.nl)
		curr_node_inds = []
		curr_node_dists = []
		for r,row in enumerate(dists):
			sortedargs = np.argsort(row)[:8]
			curr_node_inds.append(sortedargs)
			curr_node_dists.append([row[arg] for arg in sortedargs])
		
		#[row.argmin() for row in cdist(self.reshape_locations(locs),self.nl)]
		return(curr_node_inds,curr_node_dists)
	
	
	def instantiate_multi_pole_dict(self):
		'''
		class method that builds an empty dictionary to contain currents corresponding to each circuit node
		returns:dictionary
		'''
		self.multi_pole_sources = dict(zip(range(len(self.nl)),[np.zeros(401) for nd in self.nl]))
	
	def return_proportional_current(self,currents,dists):
		'''
		class method that finds the relative distance of each node from the current source found inside an element and returns the proportion of the current which is then assigned to these nodes
		currents={}
		dists=list of current source-element node distances
		returns:list of np.arrays
		'''
		return([np.array(currents)*(dist/float(sum(dists))) for dist in dists])
	
	def split_cell_curr_to_poles(self,currents,locations):
		'''
		class method that looks up the current amplitude of a given location and adds it to the nodal currents to which it was assigned
		currents={}
		locations=[[x,y,z],[x1,y1,z1],...] current source locations
		'''
		if self.geo=='tet':
			curr_node_inds,curr_node_dists = self.build_curr_node_inds_tet(locations)
		
		else:
			curr_node_inds,curr_node_dists = self.build_curr_node_inds_hex(locations)
		
		nodes_to_loop = np.array(list(curr_node_inds))
		for e,elnodes in enumerate(curr_node_inds):
			proportional_currents = self.return_proportional_current(currents[e],curr_node_dists[e])
			for n,nd in enumerate(elnodes):
				self.multi_pole_sources[nd]=self.multi_pole_sources[nd]+proportional_currents[n]
	
	
	def handle_dfile(self,findex):
		'''
		class method to load currents and locations and assign cell current sources to mesh nodes
		findex=integer identifying an file index in a list of data files
		'''
		with open(self.datadir+str(findex)+'_locations_single','rb') as f:
			locations = pickle.load(f,encoding='latin1')
		with open(self.datadir+str(findex)+'_currents','rb') as f:
			currents = pickle.load(f,encoding='latin1')
		
		for key in locations.keys():
			self.split_cell_curr_to_poles(currents[key],locations[key])
	
	
	def thread_dfiles(self):
		'''
		class method that multithreads the processing of current and location files ultimately resulting in assigning of current sources to mesh nodes
		'''
		timer = timeit('threaded all files in:')
		pool = ThreadPool(self.numthreads)
		pool.map(self.handle_dfile,range(self.dfile_count))
		timer.timeout()
		

