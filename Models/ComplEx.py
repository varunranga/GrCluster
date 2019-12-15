import tensorflow
import pickle
import numpy

class ComplEx():

	def __init__(self, config):

		self.config = config
		self.train_loss_history = []

		self.entity1_embedding_vectors = tensorflow.get_variable(name = "entity_embeddings_1", shape = [config['entity_count'], config['embedding_size']])
		self.relation1_embedding_vectors = tensorflow.get_variable(name = "relation_embeddings_1", shape = [config['relation_count'], config['embedding_size']])

		self.entity2_embedding_vectors = tensorflow.get_variable(name = "entity_embeddings_2", shape = [config['entity_count'], config['embedding_size']])
		self.relation2_embedding_vectors = tensorflow.get_variable(name = "relation_embeddings_2", shape = [config['relation_count'], config['embedding_size']])

		self.head_id = tensorflow.placeholder(tensorflow.int32)
		self.tail_id = tensorflow.placeholder(tensorflow.int32)
		self.relation_id = tensorflow.placeholder(tensorflow.int32)

		self.positive_relation_id = tensorflow.placeholder(tensorflow.int32)

		self.label = tensorflow.placeholder(tensorflow.float32)
		
		self.positive_distances = tensorflow.placeholder(tensorflow.float32)

		head1_embedding_vector = tensorflow.nn.embedding_lookup(self.entity1_embedding_vectors, self.head_id)
		head2_embedding_vector = tensorflow.nn.embedding_lookup(self.entity2_embedding_vectors, self.head_id)
		tail1_embedding_vector = tensorflow.nn.embedding_lookup(self.entity1_embedding_vectors, self.tail_id)
		tail2_embedding_vector = tensorflow.nn.embedding_lookup(self.entity2_embedding_vectors, self.tail_id)
		relation1_embedding_vector = tensorflow.nn.embedding_lookup(self.relation1_embedding_vectors, self.relation_id)
		relation2_embedding_vector = tensorflow.nn.embedding_lookup(self.relation2_embedding_vectors, self.relation_id)

		positive_relation1_embedding_vector = tensorflow.nn.embedding_lookup(self.relation1_embedding_vectors, self.positive_relation_id)
		positive_relation2_embedding_vector = tensorflow.nn.embedding_lookup(self.relation2_embedding_vectors, self.positive_relation_id)

		res = tensorflow.reduce_sum((head1_embedding_vector * tail1_embedding_vector * relation1_embedding_vector) + (head2_embedding_vector * tail2_embedding_vector * relation1_embedding_vector) + (head1_embedding_vector * tail2_embedding_vector * relation2_embedding_vector) - (head2_embedding_vector * tail1_embedding_vector * relation2_embedding_vector), 1, keep_dims = False)

		self.predict = -tensorflow.reduce_sum((head1_embedding_vector * tail1_embedding_vector * relation1_embedding_vector) + (head2_embedding_vector * tail2_embedding_vector * relation1_embedding_vector) + (head1_embedding_vector * tail2_embedding_vector * relation2_embedding_vector) - (head2_embedding_vector * tail1_embedding_vector * relation2_embedding_vector), 1, keep_dims = True)

		loss_function = tensorflow.reduce_mean(tensorflow.nn.softplus(-self.label * res), 0, keep_dims = False)
		regularization_function = tensorflow.reduce_mean(head1_embedding_vector ** 2) + tensorflow.reduce_mean(tail1_embedding_vector ** 2) + tensorflow.reduce_mean(relation1_embedding_vector ** 2) + tensorflow.reduce_mean(head2_embedding_vector ** 2) + tensorflow.reduce_mean(tail2_embedding_vector ** 2) + tensorflow.reduce_mean(relation2_embedding_vector ** 2)
		
		if config['discount_factor'] == None:
			self.discount_factor = tensorflow.get_variable(name = "discount_factor", shape = [1,])
			cluster1 = tensorflow.reduce_mean(tensorflow.maximum(tensorflow.reduce_mean(tensorflow.nn.l2_normalize(positive_relation1_embedding_vector, 1)) - (1 - tensorflow.pow(self.discount_factor, self.positive_distances)), 0))
			cluster2 = tensorflow.reduce_mean(tensorflow.maximum(tensorflow.reduce_mean(tensorflow.nn.l2_normalize(positive_relation2_embedding_vector, 1)) - (1 - tensorflow.pow(self.discount_factor, self.positive_distances)), 0))
		else:		
			cluster1 = tensorflow.reduce_mean(tensorflow.maximum(tensorflow.reduce_mean(tensorflow.nn.l2_normalize(positive_relation1_embedding_vector, 1)) - (1 - tensorflow.pow(config['discount_factor'], self.positive_distances)), 0))
			cluster2 = tensorflow.reduce_mean(tensorflow.maximum(tensorflow.reduce_mean(tensorflow.nn.l2_normalize(positive_relation2_embedding_vector, 1)) - (1 - tensorflow.pow(config['discount_factor'], self.positive_distances)), 0))
		
		if config['original']:
			self.loss = loss_function + (0.0001 * regularization_function) 
		else:
			self.loss = loss_function + (0.0001 * regularization_function) + cluster1 + cluster2

		self.global_step = tensorflow.Variable(0, name = "global_step", trainable = False)
		self.optimizer = tensorflow.train.AdamOptimizer(learning_rate = config['learning_rate'])
		self.gradients = self.optimizer.compute_gradients(self.loss)
		self.learn = self.optimizer.apply_gradients(self.gradients, global_step = self.global_step)

	def save_model(self, file_name):

		dct = 	{
					'config': self.config,
					'global_step': self.global_step.eval(),
					'entity_embeddings_1': self.entity1_embedding_vectors.eval(),
					'relation_embeddings_1': self.relation1_embedding_vectors.eval(),
					'entity_embeddings_2': self.entity2_embedding_vectors.eval(),
					'relation_embeddings_2': self.relation2_embedding_vectors.eval(),
					'train_loss_history': numpy.array(self.train_loss_history)
				}

		if self.config['discount_factor'] == None:
			dct['discount_factor'] = self.discount_factor.eval()

		fileObject = open(file_name, "wb")
		pickle.dump(dct, fileObject)
		fileObject.close()

	def load_model(self, file_name):

		fileObject = open(file_name, "rb")		
		dct = pickle.load(fileObject)
		fileObject.close()

		self.config = dct['config']
		self.train_loss_history = dct['train_loss_history'].tolist()
		tensorflow.assign(self.global_step, dct['global_step']).eval()
		tensorflow.assign(self.entity1_embedding_vectors, dct['entity_embeddings_1']).eval()
		tensorflow.assign(self.relation1_embedding_vectors, dct['relation_embeddings_1']).eval()
		tensorflow.assign(self.entity2_embedding_vectors, dct['entity_embeddings_2']).eval()
		tensorflow.assign(self.relation2_embedding_vectors, dct['relation_embeddings_2']).eval()
		
		if self.config['discount_factor'] == None:
			tensorflow.assign(self.discount_factor, dct['discount_factor']).eval()

	def train_model(self, session, positive_heads, positive_tails, positive_relations, positive_distances, negative_heads, negative_tails, negative_relations):

		heads = numpy.concatenate([positive_heads, negative_heads], axis = 0)
		tails = numpy.concatenate([positive_tails, negative_tails], axis = 0)
		relations = numpy.concatenate([positive_relations, negative_relations], axis = 0)
		labels = numpy.concatenate([numpy.ones(len(positive_heads)), -1 * numpy.ones(len(positive_heads))], axis = 0)

		feed_dict = {
						self.head_id: heads,
						self.tail_id: tails,
						self.relation_id: relations,
						self.positive_relation_id: positive_relations,
						self.positive_distances: positive_distances,
						self.label: labels
					}

		_, step, loss = session.run([self.learn, self.global_step, self.loss], feed_dict = feed_dict)

		self.train_loss_history.append(loss)

		return loss

	def test_model(self, session, positive_heads, positive_tails, positive_relations):

		feed_dict = {
						self.head_id: positive_heads,
						self.tail_id: positive_tails,
						self.relation_id: positive_relations
					}

		loss = session.run(self.predict, feed_dict = feed_dict)
		loss = numpy.reshape(loss, (-1,))

		return loss