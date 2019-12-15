import numpy
import os
import pickle
from tqdm import tqdm
from itertools import chain as _chain

import sys
sys.setrecursionlimit(1000000)

class WN_HIERARCHY():

	def __init__(self):

		self.relation_count = 1
	
		if os.path.exists("wn_hierarchy_data.npy"):
			self.data = numpy.load("wn_hierarchy_data.npy")
			self.train_data = numpy.load("wn_hierarchy_train_data.npy")
			self.valid_data = numpy.load("wn_hierarchy_valid_data.npy")
			self.test_data = numpy.load("wn_hierarchy_test_data.npy")
			self.entity_count, self.id2entity, self.entity2id = pickle.load(open("wn_hierarchy_metadata.pickle", "rb"))
		else:
			self.data = list(map(lambda x: list(map(str, x.strip().split('\t'))), open("WN_HIERARCHY/noun_closure.tsv").readlines()))
			self.id2entity = dict(enumerate(sorted(list(set(list(_chain(*self.data)))))))
			self.entity2id = {v: k for k, v in self.id2entity.items()}
			self.entity_count = len(self.id2entity)
			self.data = numpy.array(list(map(lambda x: [self.entity2id[x[0]], self.entity2id[x[1]], 0], self.data)))
			numpy.random.shuffle(self.data)
			self.train_data = self.data[:-50000]
			self.valid_data = self.data[-50000:-25000]
			self.test_data = self.data[-25000:]
			numpy.save("wn_hierarchy_data", self.data)
			numpy.save("wn_hierarchy_train_data", self.train_data)
			numpy.save("wn_hierarchy_valid_data", self.valid_data)
			numpy.save("wn_hierarchy_test_data", self.test_data)
			pickle.dump((self.entity_count, self.id2entity, self.entity2id), open("wn_hierarchy_metadata.pickle", "wb"))

		if os.path.exists("wn_hierarchy_all_data.pickle"):
			self.all_data = pickle.load(open("wn_hierarchy_all_data.pickle", "rb"))
		else:
			self.all_data = set(map(tuple, self.data.tolist()))
			pickle.dump(self.all_data, open("wn_hierarchy_all_data.pickle", "wb"))

		if os.path.exists("wn_hierarchy_hierarchy.pickle"):
			self.hierarchy = pickle.load(open("wn_hierarchy_hierarchy.pickle", "rb"))
		else:
			self.hierarchy = self.create_hierarchy()
			pickle.dump(self.hierarchy, open("wn_hierarchy_hierarchy.pickle", "wb"))
			del self._children_of_all_entities

		if os.path.exists("wn_hierarchy_entity_distances.npy"):
			self.entity_distances = numpy.load("wn_hierarchy_entity_distances.npy")
		else:
			self.entity_distances = self.get_entity_distances()
			numpy.save("wn_hierarchy_entity_distances", self.entity_distances)
		
	def create_hierarchy(self):

		self._children_of_all_entities = {i:list() for i in range(self.entity_count)}

		data = self.data
	
		for head_id, tail_id, relation_id in data:
			self._children_of_all_entities[tail_id].append(head_id)

		return self._children_of_all_entities

	def get_entity_distances(self):

		def recursive_get_children_with_depth(children):
			dct = {child: 1 for child in children}
			for child in children:
				if child in _distances:
					_dct = _distances[child].copy()
					for key in _dct:
						dct[key] = _dct[key] + 1
				else:
					_distances[child] = recursive_get_children_with_depth(self.hierarchy[child])
					_dct = _distances[child].copy()
					for key in _dct:
						dct[key] = _dct[key] + 1
			return dct

		data = self.train_data
		root_entities = self.id2entity.keys()
		hierarchy = self.hierarchy
	
		_distances = {}

		for parent_entity in tqdm(range(self.entity_count)):
			_distances[parent_entity] = recursive_get_children_with_depth(self.hierarchy[parent_entity])			

		distances = []

		for head_id, tail_id, relation_id in tqdm(data):
			distances.append(_distances[tail_id][head_id])

		return numpy.array(distances)

	def create_train_data_generator(self, batch_size = 512, sampling_type = 'uniform'):
		
		data = self.train_data
		distances = self.entity_distances
			
		for i in range(0, len(data), batch_size):
			positive_heads = data[i:i+batch_size, 0]
			positive_tails = data[i:i+batch_size, 1]
			positive_relations = data[i:i+batch_size, 2]
			positive_distances = distances[i:i+batch_size]

			negative_heads = positive_heads.copy()
			negative_relations = positive_relations.copy()
			negative_tails = positive_tails.copy()

			for j in range(len(positive_heads)):
				if numpy.random.rand() < 0.5:
					negative_heads[j] = numpy.random.randint(0, self.entity_count)
				else:
					negative_tails[j] = numpy.random.randint(0, self.entity_count)

			yield positive_heads, positive_tails, positive_relations, positive_distances, negative_heads, negative_tails, negative_relations

	def create_positive_generator_for_triplet_classification(self, batch_size = 512):
		
		data = self.test_data
	
		for i in range(0, len(data), batch_size):
			positive_heads = data[i:i+batch_size, 0]
			positive_tails = data[i:i+batch_size, 1]
			positive_relations = data[i:i+batch_size, 2]

			yield positive_heads, positive_tails, positive_relations

	def create_negative_generator_for_triplet_classification(self, batch_size = 512):
		
		data = self.test_data
	
		for i in range(0, len(data), batch_size):
			positive_heads = data[i:i+batch_size, 0]
			positive_tails = data[i:i+batch_size, 1]
			positive_relations = data[i:i+batch_size, 2]

			negative_heads = positive_heads.copy()
			negative_relations = positive_relations.copy()
			negative_tails = positive_tails.copy()


			for j in range(len(positive_heads)):
				if numpy.random.rand() < 0.5:
					negative_heads[j] = numpy.random.randint(0, self.entity_count)
				else:
					negative_tails[j] = numpy.random.randint(0, self.entity_count)

			yield negative_heads, negative_tails, negative_relations


	def create_test_data_generator(self, batch_size = 512):
			
		data = self.test_data
		
		for i in range(0, len(data), batch_size):
			positive_heads = data[i:i+batch_size, 0]
			positive_tails = data[i:i+batch_size, 1]
			positive_relations = data[i:i+batch_size, 2]

			yield positive_heads, positive_tails, positive_relations

if __name__ == '__main__':
	w = WN_HIERARCHY()