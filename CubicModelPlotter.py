import os
import pickle
import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import pandas as pd



class PlotModelSimulation():
	
	
	'''
	
	PlotModelSimulation class is used to visualize using mayavi/vtk a series of mesh solutions that can later be stitched together into a video or .gif using any tool the user prefers.
	
	nodes = the [[x1,x2,...],[y1,y2,...],[z1,z2,...]] points of the mesh as constructed by CubicNodes or UnstructuredNodes (ordered)
	tsteps = the range of timesteps in the series of solutions e.g. [1,2,3,4,...] as they were originally provided by the user to BuildSolution class in SolutionBuilder.py
	soldir = the directory where the '*_solution' files can be found that were created by SolutionBuilder.py
	
	Usage:
	import neurospice
	from neurospice.CubicModelPlotter import PlotModelSimulation
	#after solving 100 timesteps and having *_solution placed in the current working directory
	plotter = PlotModelSimulation(nodes,range(100),soldir=os.getcwd())
	'''
	
	
	def __init__(self,nodes,tsteps,soldir=os.getcwd()):
		self.nodes = nodes
		self.minv = 0
		self.maxv = 0
		solution_fs = [fname for fname in os.listdir(soldir) if '_solution' in fname]
		for fname in solution_fs:
			with open(fname,'rb') as f:
				sol = pickle.load(f,encoding='latin1')
				if np.min(list(sol.nvs.values())) < self.minv:
					self.minv = np.min(list(sol.nvs.values()))
				if np.max(list(sol.nvs.values())) > self.maxv:
					self.maxv = np.max(list(sol.nvs.values()))
	
		for tstep in tsteps:
			with constant_camera_view():
				self.plot_nodal_voltages(tstep)
	
	def plot_nodal_voltages(self,tstep,az=0,el=0,dist=50,res=16,scale_f=0.45,cmap='coolwarm',pprojection=True,png_sz=(1200,900),bgc=None):
		
		'''
		
		This class method takes a time step and loads the corresponding solution and plots the nodal voltages
		
		TO-DO:__init__ needs to take these plotting arguments with defaults to provide user with flexibility in image creation
		
		tstep = value from tsteps specified in __init__ args
		az = azimuth of mayavi camera (default=0)
		el = elevation of mayavi camera (default=0)
		dist = distance of camera from actor (default=50)
		res = resolution of mlab points3d objects (default=16)
		scale_f = scale_factor of mlab points3d objects (default=0.45)
		cmap = colormap used in scalar coloring of the mlab scene (default='coolwarm')
		pprojection = parallel_projection of the mlab scene (default=True)
		png_sz = mlab figure size in px
		bgc = background color (bgcolor) of mlab figure (default=None (i.e. transparent))
		
		'''
		
		with open(str(tstep)+'_solution','rb') as f:
			sol = pickle.load(f,encoding='latin1')
		
		x = []
		y = []
		z = []
		nv = []
		for key in sol.nvs.keys():
			x.append(self.nodes.xs[key])
			y.append(self.nodes.ys[key])
			z.append(self.nodes.zs[key])
			nv.append(np.array(sol.nvs[key]))
		
		mlab.figure(bgcolor=bgc, size=png_sz)
		mlab.points3d(x,y,z,nv,colormap=cmap,resolution=res,scale_factor=scale_f,scale_mode='scalar')
		if pprojection:
			mlab.gcf().scene.parallel_projection = True
		else:
			mlab.gcf().scene.parallel_projection = False
		
		mlab.view(azimuth=az, elevation=el, distance=dist)
		mlab.savefig(str(tstep)+'.png')
		mlab.close(all=True)
		print('done with '+str(tstep))


class constant_camera_view(object):
	
	'''
	
	Helper class used to ensure the vtk camera is always reset between renderings
	
	'''
	
	
	def __init__(self):
		pass
	
	def __enter__(self):
		self.orig_no_render = mlab.gcf().scene.disable_render
		if not self.orig_no_render:
			mlab.gcf().scene.disable_render = True
		cc = mlab.gcf().scene.camera
		self.orig_pos = cc.position
		self.orig_fp = cc.focal_point
		self.orig_view_angle = cc.view_angle
		self.orig_view_up = cc.view_up
		self.orig_clipping_range = cc.clipping_range
	
	def __exit__(self, t, val, trace):
		cc = mlab.gcf().scene.camera
		cc.position = self.orig_pos
		cc.focal_point = self.orig_fp
		cc.view_angle =  self.orig_view_angle 
		cc.view_up = self.orig_view_up
		cc.clipping_range = self.orig_clipping_range
	
		if not self.orig_no_render:
			mlab.gcf().scene.disable_render = False
		if t != None:
			print (t, val, trace)
			#ipdb.post_mortem(trace)



class PlotVoltageTraces():
	
	'''
	
	PlotVoltageTraces class is used to identify the mesh element that contains a user-designated point of interest and interpolate nodal voltages to that point within the mesh to provide a best estimate field potential. If a series of solutions were generated then all *_solution files are loaded in series to yield a time-series of tri-linearly interpolated nodal voltage series.
	
	nodes = the [[x1,x2,...],[y1,y2,...],[z1,z2,...]] points of the mesh as constructed by CubicNodes or UnstructuredNodes (ordered)
	geo = the geometry of the mesh that was solved. Acceptable arguments include 'tet','hex'. (default='tet')
	name = unique name that can be used to label output (default=None)
	solfs_dir = directory where *_solution files can be found (default=os.getcwd())
	
	Usage:
	import neurospice
	from neurospice.CubicModelPlotter import PlotVoltageTraces
	#after solving 100 timesteps and having *_solution placed in the current working directory
	plotter = PlotVoltageTraces(nodes,geo='tet',name=unique_name,solfs_dir=os.getcwd())
	plotter.interpolated_recording_to_csv(point=[1,1,1])
	'''
	
	
	def __init__(self,nodes,geo='tet',name=None,solfs_dir=os.getcwd()):
		if geo not in ['tet','hex']:
			print('User designated geometry not a valid option. Choose from "tet","hex"')
			return()
		
		self.solfs_dir=solfs_dir
		self.geo=geo
		self.nodes = nodes
		self.name=name
		self.nv_series = self.load_nv_series()
	
	def load_nv_series(self):
	
	
		'''
		
		class method that loads *_solution files from solfs_dir arg provided in __init__
		returns:{nodeids:voltage timeseries}
		
		'''
		
		
		solution_fs = [fname for fname in os.listdir(self.solfs_dir) if '_solution' in fname]
		solution_fs = sorted(solution_fs)
		with open(self.solfs_dir+'/'+solution_fs[0],'rb') as f:
			sol = pickle.load(f,encoding='latin1')
			nv_series = {}
			for key in sol.nvs.keys():
				nv_series[key] = [sol.nvs[key]]
		
		for fname in solution_fs[1:]:
			with open(self.solfs_dir+'/'+fname,'rb') as f:
				sol = pickle.load(f,encoding='latin1')
			
			for key in sol.nvs.keys():
				nv_series[key].append(sol.nvs[key])
		
		return(nv_series)
	
	
	def find_recording_node(self,recording_location,xs,ys,zs):
		
		'''
		
		method finds the nearest mesh node to a user specified point of interest
		
		recording_location = user designated point of interest [x,y,z] where x,y,z are floats or integers that are contained by the bounding box of the mesh
		xs, ys, zs = node lists as created by CubicNodes or UnstructuredNodes
		returns:integer, index of nearest node
		'''
		
		return(cdist([recording_location], np.array(list(zip(xs,ys,zs)))).argmin())
	
	def interpolate_recording_node(self,recording_location,xs,ys,zs):
		
		'''
		
		method finds the set of nodes that form the vertices of the mesh element that contains a user specified point of interest. This method returns that point of interest to vertice distance weighted scalars of each element vertices that are later used to split currents into each element node
		
		recording_location = user designated point of interest [x,y,z] where x,y,z are floats or integers that are contained by the bounding box of the mesh
		xs, ys, zs = node lists as created by CubicNodes or UnstructuredNodes
		returns:list of lists of floats of current amplitudes for nodes of an element
		'''
		
		dists = cdist([recording_location], np.array(list(zip(xs,ys,zs))))
		if self.geo=='tet':
			nearest_node_inds = np.argsort(dists)[0][:4]

		else:
			nearest_node_inds = np.argsort(dists)[0][:8]
		
		nearest_node_props = np.array([dists[0][n] for n in nearest_node_inds])
		nearest_node_props = nearest_node_props/sum(nearest_node_props)
		
		return(np.mean(np.array([np.array(self.nv_series[node_ind])*nearest_node_props[n] for n,node_ind in enumerate(nearest_node_inds)]),axis=0))
	
	def interpolated_recording_to_csv(self,points=[[935.84,243.9,120.1]]):
		
		
		'''
		
		method finds the set of nodes that form the vertices of the mesh element that contains a user specified point of interest. This method returns that point of interest to vertice distance weighted scalars of each element vertices that are later used to split currents into each element node
		It then outputs the series of results to a csv file in the same directory as the solfs_dir
		
		point = recording_location, user designated point of interest [x,y,z] where x,y,z are floats or integers that are contained by the bounding box of the mesh (default=[1068.0,372.0,480.0] is used in tests)
		
		'''
		
		interested_series = []
		for point in points:
			nv = self.interpolate_recording_node(point,self.nodes.xs,self.nodes.ys,self.nodes.zs)
			interested_series.append(np.array(nv[:74]))
		
		if self.name == None:
			target = self.solfs_dir+'/'+self.geo+'_'+str(len(self.nodes.xs))+'.csv'
		else:
			target = self.solfs_dir+'/'+self.name+'_'+self.geo+'_'+str(len(self.nodes.xs))+'.csv'
		
		result = pd.DataFrame()
		for s,ser in enumerate(interested_series):
			result[s] = ser 
		
		result.to_csv(target)
