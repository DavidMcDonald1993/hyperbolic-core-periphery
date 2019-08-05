import numpy as np

class CoreTree(object):
	
	def __init__(self, root, subsets, cores, depth):
		self.root = root
		self.cores = cores
		self.depth = depth

		if len(subsets) > 0:
			self.children = [CoreTree(child, 
				list(filter(lambda subset: cores[subset] < cores[child], subsets)), 
				self.cores,
				self.depth + 1)
				for child in (c1 for c1 in subsets if not any((cores[c1] < cores[c2] for c2 in subsets)))]
		else: 
			self.children = []
	
	def __len__(self):
		return 1 + sum((len(child) for child in self.children))
	
	def __str__(self):
		s = "|" * (self.depth - 1) + "-" * int(self.depth > 0)
		# s += "{}\n".format(self.cores[self.root])
		s += "{} {}\n".format(self.root, len(self.cores[self.root]))
		for child in self.children:
			s += "{} ".format(len(self.cores[self.root] - self.cores[child.root]))
		s += "\n"
		for child in self.children:
			s += "{}".format(child)
		return s

	def __eq__(self, other):
		if isinstance(other, self.__class__):
			return (
				self.root == other.root 
				and self.subsets == other.subsets
				and self.cores == other.cores
				and self.depth == other.depth
				)
		return False

	def __ne__(self, other):
		return not self == other

	