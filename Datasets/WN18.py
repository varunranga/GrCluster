import numpy
import os
import pickle
from tqdm import tqdm

import sys
sys.setrecursionlimit(1000000)

class WN18():

	def __init__(self):

		self.entity_count = 40943
		self.relation_count = 18
	
		if os.path.exists("wn18_train_data.npy"):
			self.train_data = numpy.load("wn18_train_data.npy")
		else:
			self.train_data = numpy.array(list(map(lambda x: numpy.array(list(map(int, x.split()))), open("WN18/train2id.txt").readlines()[1:]))) 
			numpy.save("wn18_train_data", self.train_data)

		if os.path.exists("wn18_valid_data.npy"):
			self.valid_data = numpy.load("wn18_valid_data.npy")
		else:
			self.valid_data = numpy.array(list(map(lambda x: numpy.array(list(map(int, x.split()))), open("WN18/valid2id.txt").readlines()[1:]))) 
			numpy.save("wn18_valid_data", self.valid_data)

		if os.path.exists("wn18_test_data.npy"):
			self.test_data = numpy.load("wn18_test_data.npy")
		else:
			self.test_data = numpy.array(list(map(lambda x: numpy.array(list(map(int, x.split()))), open("WN18/test2id.txt").readlines()[1:]))) 
			numpy.save("wn18_test_data", self.test_data)

		if os.path.exists("wn18_all_data.pickle"):
			self.all_data = pickle.load(open("wn18_all_data.pickle", "rb"))
		else:
			self.all_data = set(map(tuple, self.train_data.tolist() + self.valid_data.tolist() + self.test_data.tolist()))
			pickle.dump(self.all_data, open("wn18_all_data.pickle", "wb"))

		if os.path.exists("wn18_one_to_one_data.npy"):
			self.one_to_one_data = numpy.load("wn18_one_to_one_data.npy")
		else:
			self.one_to_one_data = numpy.array(list(map(lambda x: numpy.array(list(map(int, x.split()))), open("WN18/1-1.txt").readlines()[1:]))) 
			numpy.save("wn18_one_to_one_data", self.one_to_one_data)

		if os.path.exists("wn18_one_to_many_data.npy"):
			self.one_to_many_data = numpy.load("wn18_one_to_many_data.npy")
		else:
			self.one_to_many_data = numpy.array(list(map(lambda x: numpy.array(list(map(int, x.split()))), open("WN18/1-n.txt").readlines()[1:]))) 
			numpy.save("wn18_one_to_many_data", self.one_to_many_data)

		if os.path.exists("wn18_many_to_one_data.npy"):
			self.many_to_one_data = numpy.load("wn18_many_to_one_data.npy")
		else:
			self.many_to_one_data = numpy.array(list(map(lambda x: numpy.array(list(map(int, x.split()))), open("WN18/n-1.txt").readlines()[1:]))) 
			numpy.save("wn18_many_to_one_data", self.many_to_one_data)

		if os.path.exists("wn18_many_to_many_data.npy"):
			self.many_to_many_data = numpy.load("wn18_many_to_many_data.npy")
		else:
			self.many_to_many_data = numpy.array(list(map(lambda x: numpy.array(list(map(int, x.split()))), open("WN18/n-n.txt").readlines()[1:]))) 
			numpy.save("wn18_many_to_many_data", self.many_to_many_data)		

		if os.path.exists("wn18_relation2id.pickle"):
			self.relation2id = pickle.load(open("wn18_relation2id.pickle", "rb"))
		else:
			self.relation2id = dict(map(lambda x: tuple([x.split()[0], int(x.split()[1])]), open("WN18/relation2id.txt").readlines()[1:]))
			pickle.dump(self.relation2id, open("wn18_relation2id.pickle", "wb"))

		if os.path.exists("wn18_train_hierarchy.pickle"):
			self.train_hierarchy = pickle.load(open("wn18_train_hierarchy.pickle", "rb"))
			self.train_root_entities = pickle.load(open("wn18_train_root_entities.pickle", "rb"))
		else:
			self.train_root_entities, self.train_hierarchy = self.create_hierarchy(type_of_split = 'train')
			pickle.dump(self.train_hierarchy, open("wn18_train_hierarchy.pickle", "wb"))
			pickle.dump(self.train_root_entities, open("wn18_train_root_entities.pickle", "wb"))
			del self._children_of_all_entities

		if os.path.exists("wn18_valid_hierarchy.pickle"):
			self.valid_hierarchy = pickle.load(open("wn18_valid_hierarchy.pickle", "rb"))
			self.valid_root_entities = pickle.load(open("wn18_valid_root_entities.pickle", "rb"))
		else:
			self.valid_root_entities, self.valid_hierarchy = self.create_hierarchy(type_of_split = 'valid')
			pickle.dump(self.valid_hierarchy, open("wn18_valid_hierarchy.pickle", "wb"))
			pickle.dump(self.valid_root_entities, open("wn18_valid_root_entities.pickle", "wb"))
			del self._children_of_all_entities

		if os.path.exists("wn18_train_entity_distances.npy"):
			self.train_entity_distances = numpy.load("wn18_train_entity_distances.npy")
		else:
			self.train_entity_distances = self.get_entity_distances(type_of_split = 'train')
			numpy.save("wn18_train_entity_distances", self.train_entity_distances)

		if os.path.exists("wn18_valid_entity_distances.npy"):
			self.valid_entity_distances = numpy.load("wn18_valid_entity_distances.npy")
		else:
			self.valid_entity_distances = self.get_entity_distances(type_of_split = 'valid')
			numpy.save("wn18_valid_entity_distances", self.valid_entity_distances)

		if os.path.exists("wn18_bernoulli_probabilities.pickle"):
			self.bernoulli_probablities = pickle.load(open("wn18_bernoulli_probabilities.pickle", "rb"))
		else:
			self.bernoulli_probablities = {}
			for i in range(self.relation_count):
				data = self.train_data[self.train_data[:, 2] == i]
				head_entities = set(data[:, 0])
				tph = sum(len(data[data[:, 0] == head_entity]) for head_entity in head_entities) / len(head_entities)
				tail_entities = set(data[:, 1])
				hpt = sum(len(data[data[:, 1] == tail_entity]) for tail_entity in tail_entities) / len(tail_entities)
				self.bernoulli_probablities[i] = tph / (hpt + tph)
			pickle.dump(self.bernoulli_probablities, open("wn18_bernoulli_probabilities.pickle", "wb"))

	def create_hierarchy(self, type_of_split = 'train'):

		def recursive_insert(parent_entity, child_entity):

			if (self._children_of_all_entities[parent_entity] != None) and (len(self._children_of_all_entities[parent_entity]) == 0):
				return False

			for _child_entity in self._children_of_all_entities[parent_entity]:
				if _child_entity == child_entity:
					self._children_of_all_entities[parent_entity][_child_entity] = self._children_of_all_entities[child_entity]
					return True
				else:
					if recursive_insert(_child_entity, child_entity):
						return True

			return False

		self._children_of_all_entities = {i:{} for i in range(self.entity_count)}

		if type_of_split == 'train':
			data = self.train_data
		elif type_of_split == 'valid':
			data = self.valid_data

		for head_id, tail_id, relation_id in data:
			if relation_id == self.relation2id['_hypernym']:
				if head_id not in self._children_of_all_entities[tail_id]:
					self._children_of_all_entities[tail_id][head_id] = {}
			elif relation_id == self.relation2id['_hyponym']:
				if tail_id not in self._children_of_all_entities[head_id]:
					self._children_of_all_entities[head_id][tail_id] = {}

		children = []
		bar_iterator = tqdm(range(self.entity_count ** 2)).__iter__()

		for entity_id in range(self.entity_count):
			for _entity_id in range(self.entity_count):
				if (not _entity_id == entity_id) and recursive_insert(_entity_id, entity_id):
					children.append(entity_id)
				bar_iterator.__next__()

		root_entities = set(list(range(self.entity_count))) - set(children)

		return root_entities, self._children_of_all_entities

	def get_entity_distances(self, type_of_split = 'train'):

		def recursive_get_chain(entity, parent_entity = None, parent_dictionary = None):

			if parent_entity == None:
				for parent_entity in root_entities:
					chain = recursive_get_chain(entity, parent_entity, hierarchy[parent_entity])
					if chain:
						return max(chain, key = len)
			elif entity == parent_entity:
				return [[entity]]
			else:
				for child_entity in parent_dictionary:
					chain = recursive_get_chain(entity, child_entity, parent_dictionary[child_entity])
					if chain:
						return [[parent_entity] + max(chain, key = len)]
			
			return False

		if type_of_split == 'train':
			data = self.train_data
			root_entities = self.train_root_entities
			hierarchy = self.train_hierarchy
		elif type_of_split == 'valid':
			data = self.valid_data
			root_entities = self.valid_root_entities
			hierarchy = self.valid_hierarchy

		distances = []

		for head_id, tail_id, relation_id in tqdm(data):
			chain_to_head = recursive_get_chain(head_id)
			chain_to_tail = recursive_get_chain(tail_id)
			distance = len((set(chain_to_head) | set(chain_to_tail)) - (set(chain_to_head) & set(chain_to_tail)))
			distances.append(distance)

		return numpy.array(distances)

	def create_train_data_generator(self, batch_size = 512, sampling_type = 'uniform'):
		
		data = self.train_data
		distances = self.train_entity_distances

		if sampling_type == 'uniform':
			
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

		elif sampling_type == 'bernoulli':
			
			for i in range(0, len(data), batch_size):
				positive_heads = data[i:i+batch_size, 0]
				positive_tails = data[i:i+batch_size, 1]
				positive_relations = data[i:i+batch_size, 2]
				positive_distances = distances[i:i+batch_size]

				negative_heads = positive_heads.copy()
				negative_relations = positive_relations.copy()
				negative_tails = positive_tails.copy()

				for j in range(len(positive_heads)):

					if numpy.random.rand() < self.bernoulli_probablities[negative_relations[j]]:
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

	def create_test_data_generator(self, batch_size = 512, type_of_split = 'test'):
			
		if type_of_split == 'valid':
			data = self.valid_data
		elif type_of_split == 'test':
			data = self.test_data
		elif type_of_split == 'one_to_one':
			data = self.one_to_one_data
		elif type_of_split == 'one_to_many':
			data = self.one_to_many_data
		elif type_of_split == 'many_to_one':
			data = self.many_to_one_data
		elif type_of_split == 'many_to_many':
			data = self.many_to_many_data

		for i in range(0, len(data), batch_size):
			positive_heads = data[i:i+batch_size, 0]
			positive_tails = data[i:i+batch_size, 1]
			positive_relations = data[i:i+batch_size, 2]

			yield positive_heads, positive_tails, positive_relations

if __name__ == '__main__':
	w = WN18()
	pickle.dump(w, open("WN18.pickle", "wb"))