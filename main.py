import tensorflow
import argparse
import numpy
import pickle
from Models.TransE import TransE
from Models.TransH import TransH
from Models.TransR import TransR
from Models.TransD import TransD
from Models.HolE import HolE
from Models.DistMult import DistMult
from Models.ComplEx import ComplEx
from Datasets.WN18 import WN18
from Datasets.WN_HIERARCHY import WN_HIERARCHY
from tqdm import tqdm
from math import ceil
from math import inf

parser = argparse.ArgumentParser()
parser.add_argument("--no-train", help = "Do not train embeddings", default = False, action = 'store_true')
parser.add_argument("-s", "--embedding-size", help = "Embedding size of each vector", type = int, default = 100)
parser.add_argument("-b", "--batch-size", help = "Batch size while training", type = int, default = 1024)
parser.add_argument("-m", "--margin", help = "Margin of error allowed in the loss", type = float, default = 1)
parser.add_argument("-r", "--learning-rate", help = "Learning rate for the optimizer", type = float, default = 1e-3)
parser.add_argument("-e", "--epochs", help = "Number of epochs to train embeddings", type = int, default = 500)
parser.add_argument("-t", "--infinitely-train", help = "Train infinitely with patience", default = False, action = 'store_true')
parser.add_argument("-p", "--patience", help = "Patience while training the embedding model for validation loss to improve", type = int, default = 50)
parser.add_argument("-o", "--output-file", help = "Pickle file name for the trained model to save", type = str, default = None)
parser.add_argument("-i", "--input-file", help = "Pickle file name for the trained model to load", type = str, default = None)
parser.add_argument("-d", "--dataset", help = "Dataset to be used", type = str, default = 'WN18')
parser.add_argument("-a", "--embedding-model", help = "Embedding Model to be used", type = str, default = 'TransE')
parser.add_argument("-n", "--original", help = "Train using original embedding model", default = False, action = "store_true")
parser.add_argument("-q", "--sampling-type", help = "Method used to sample data", type = str, default = 'uniform')
parser.add_argument("-f", "--discount-factor", help = "Discounting factor used for distance", type = float, default = None)
parser.add_argument("--no-test", help = "Do not test embeddings", default = False, action = 'store_true')
parser.add_argument("--no-link-prediction", help = "Do not test embeddings for link prediction", default = False, action = 'store_true')
parser.add_argument("--no-triplet-classification", help = "Do not test embeddings on triplet classification", default = False, action = 'store_true')
parser.add_argument("-g", "--test-setting", help = "Sampling setting while testing", type = str, default = "raw")
parser.add_argument("-c", "--triplet-classification-times", help = "Number of times triplet classification must be performed", type = int, default = 25)
args = parser.parse_args()

dataset = eval(args.dataset)()

with tensorflow.Graph().as_default():
	
	session = tensorflow.Session()
	
	with session.as_default():

		config = 	{
						'entity_count': dataset.entity_count,
						'relation_count': dataset.relation_count,
						'embedding_size': args.embedding_size,
						'margin': args.margin,
						'learning_rate': args.learning_rate,
						'discount_factor': args.discount_factor,
						'original': args.original
					}
		
		embedding_model = eval(args.embedding_model)(config)

		session.run(tensorflow.global_variables_initializer())

		if args.input_file:
			embedding_model.load_model(file_name = args.input_file)

		if not args.no_train:

			if args.infinitely_train:

				waited = 0
				best_loss = inf
				epoch = 0

				while waited < args.patience:
					print()
					print("EPOCH", epoch + 1)
					print("-"*79)
					
					print("TRAINING")
					
					train_generator = dataset.create_train_data_generator(batch_size = args.batch_size, sampling_type = args.sampling_type).__iter__()
					
					training_step = 0
					training_loss = 0
					
					for _  in tqdm(range(ceil(len(dataset.train_data) / args.batch_size))):
						positive_heads, positive_tails, positive_relations, positive_distances, negative_heads, negative_tails, negative_relations = train_generator.__next__()
					
						training_loss += embedding_model.train_model(session, positive_heads, positive_tails, positive_relations, positive_distances, negative_heads, negative_tails, negative_relations)
						training_step += 1

					average_training_loss = training_loss / training_step
					print("Train loss:", average_training_loss)
					
					if average_training_loss < best_loss:
						best_loss = average_training_loss
						waited = 0

						if args.output_file:
							embedding_model.save_model(file_name = args.output_file)

					else:
						waited += 1

					epoch += 1

					print("-"*79)
					print()
			
			else:
				
				for epoch in range(args.epochs):
					train_generator = dataset.create_train_data_generator(batch_size = args.batch_size, sampling_type = args.sampling_type).__iter__()
					
					print()
					print("EPOCH", epoch + 1, "/", args.epochs)
					print("-"*79)
					
					print("TRAINING")
					
					training_step = 0
					training_loss = 0
					
					for _  in tqdm(range(ceil(len(dataset.train_data) / args.batch_size))):
						positive_heads, positive_tails, positive_relations, positive_distances, negative_heads, negative_tails, negative_relations = train_generator.__next__()
						
						training_loss += embedding_model.train_model(session, positive_heads, positive_tails, positive_relations, cluster, negative_heads, negative_tails, negative_relations)
						training_step += 1

					print("Train loss:", training_loss / training_step)
					
					if args.output_file:
						embedding_model.save_model(file_name = args.output_file)
					
					print("-"*79)
					print()

		if (not args.no_train) and args.output_file:
			embedding_model.load_model(file_name = args.output_file)

		if not args.no_test:

			if args.input_file:
				model = pickle.load(open(args.input_file, "rb"))
				if "TEST" in model:
					test_model = model["TEST"]
				else:
					test_model = {}
			else:
				test_model = {}

			if not args.no_link_prediction:			
		
				print("LINK PREDICTION")
				
				test_model["LINK_PREDICTION"] = {}

				def test_replace_tail(head, tail, relation):
					for i in range(0, dataset.entity_count, args.batch_size):
						yield numpy.array(list(map(lambda j: [head, i + j, relation], range(min(args.batch_size, dataset.entity_count - i)))))

				def test_replace_head(head, tail, relation):
					for i in range(0, dataset.entity_count, args.batch_size):
						yield numpy.array(list(map(lambda j: [i + j, tail, relation], range(min(args.batch_size, dataset.entity_count - i)))))

				def test_perform_link_prediction(data, replace_entity):
					ranks = []
					for head, tail, relation in tqdm(data):
						losses = []

						for test_triples in replace_entity(head, tail, relation):

							if args.test_setting == 'filter':
								test_triples = numpy.array(list(map(list, set(map(tuple, test_triples.tolist())) - dataset.all_data)))

							if len(test_triples) == 0:
								continue

							test_heads = test_triples[:, 0]
							test_tails = test_triples[:, 1]
							test_relations = test_triples[:, 2]

							losses += embedding_model.test_model(session, test_heads, test_tails, test_relations).tolist()

						if args.test_setting == 'filter':
							losses += embedding_model.test_model(session, [head], [tail], [relation]).tolist()						
							loss = losses[-1]							
						else:						
							if replace_entity == test_replace_head:
								loss = losses[head]
							elif replace_entity == test_replace_tail:
								loss = losses[tail]

						losses = sorted(losses)
						rank = losses.index(loss)
						ranks.append(rank)

						# print(rank)

					ranks = numpy.array(ranks)
					mean_rank = numpy.mean(ranks)
					hits_at_10 = len(numpy.argwhere(ranks < 10)) / len(ranks)

					return ranks, mean_rank, hits_at_10

				print("OVERALL")			
				test_model["LINK_PREDICTION"]["TEST"] = {}
				print("REPLACING HEAD ENTITY")
				test_model["LINK_PREDICTION"]["TEST"]["REPLACE_HEAD"] = {}
				ranks, mean_rank, hits_at_10 = test_perform_link_prediction(dataset.test_data, test_replace_head)
				test_model["LINK_PREDICTION"]["TEST"]["REPLACE_HEAD"]["ranks"] = ranks
				test_model["LINK_PREDICTION"]["TEST"]["REPLACE_HEAD"]["mean_rank"] = mean_rank
				test_model["LINK_PREDICTION"]["TEST"]["REPLACE_HEAD"]["hits_at_10"] = hits_at_10
				print("mean_rank:", mean_rank)
				print("hits_at_10:", hits_at_10)
				print("REPLACING TAIL ENTITY")
				test_model["LINK_PREDICTION"]["TEST"]["REPLACE_TAIL"] = {}
				ranks, mean_rank, hits_at_10 = test_perform_link_prediction(dataset.test_data, test_replace_tail)
				test_model["LINK_PREDICTION"]["TEST"]["REPLACE_TAIL"]["ranks"] = ranks
				test_model["LINK_PREDICTION"]["TEST"]["REPLACE_TAIL"]["mean_rank"] = mean_rank
				test_model["LINK_PREDICTION"]["TEST"]["REPLACE_TAIL"]["hits_at_10"] = hits_at_10
				print("mean_rank:", mean_rank)
				print("hits_at_10:", hits_at_10)

				if not (args.dataset == "WN_HIERARCHY"):

					print("ONE-TO-ONE")
					test_model["LINK_PREDICTION"]["ONE-TO-ONE"] = {}
					print("REPLACING HEAD ENTITY")
					test_model["LINK_PREDICTION"]["ONE-TO-ONE"]["REPLACE_HEAD"] = {}
					ranks, mean_rank, hits_at_10 = test_perform_link_prediction(dataset.one_to_one_data, test_replace_head)
					test_model["LINK_PREDICTION"]["ONE-TO-ONE"]["REPLACE_HEAD"]["ranks"] = ranks
					test_model["LINK_PREDICTION"]["ONE-TO-ONE"]["REPLACE_HEAD"]["mean_rank"] = mean_rank
					test_model["LINK_PREDICTION"]["ONE-TO-ONE"]["REPLACE_HEAD"]["hits_at_10"] = hits_at_10
					print("mean_rank:", mean_rank)
					print("hits_at_10:", hits_at_10)
					print("REPLACING TAIL ENTITY")
					test_model["LINK_PREDICTION"]["ONE-TO-ONE"]["REPLACE_TAIL"] = {}
					ranks, mean_rank, hits_at_10 = test_perform_link_prediction(dataset.one_to_one_data, test_replace_tail)
					test_model["LINK_PREDICTION"]["ONE-TO-ONE"]["REPLACE_TAIL"]["ranks"] = ranks
					test_model["LINK_PREDICTION"]["ONE-TO-ONE"]["REPLACE_TAIL"]["mean_rank"] = mean_rank
					test_model["LINK_PREDICTION"]["ONE-TO-ONE"]["REPLACE_TAIL"]["hits_at_10"] = hits_at_10
					print("mean_rank:", mean_rank)
					print("hits_at_10:", hits_at_10)

					print("ONE-TO-MANY")
					test_model["LINK_PREDICTION"]["ONE-TO-MANY"] = {}
					print("REPLACING HEAD ENTITY")
					test_model["LINK_PREDICTION"]["ONE-TO-MANY"]["REPLACE_HEAD"] = {}
					ranks, mean_rank, hits_at_10 = test_perform_link_prediction(dataset.one_to_many_data, test_replace_head)
					test_model["LINK_PREDICTION"]["ONE-TO-MANY"]["REPLACE_HEAD"]["ranks"] = ranks
					test_model["LINK_PREDICTION"]["ONE-TO-MANY"]["REPLACE_HEAD"]["mean_rank"] = mean_rank
					test_model["LINK_PREDICTION"]["ONE-TO-MANY"]["REPLACE_HEAD"]["hits_at_10"] = hits_at_10
					print("mean_rank:", mean_rank)
					print("hits_at_10:", hits_at_10)
					print("REPLACING TAIL ENTITY")
					test_model["LINK_PREDICTION"]["ONE-TO-MANY"]["REPLACE_TAIL"] = {}
					ranks, mean_rank, hits_at_10 = test_perform_link_prediction(dataset.one_to_many_data, test_replace_tail)
					test_model["LINK_PREDICTION"]["ONE-TO-MANY"]["REPLACE_TAIL"]["ranks"] = ranks
					test_model["LINK_PREDICTION"]["ONE-TO-MANY"]["REPLACE_TAIL"]["mean_rank"] = mean_rank
					test_model["LINK_PREDICTION"]["ONE-TO-MANY"]["REPLACE_TAIL"]["hits_at_10"] = hits_at_10
					print("mean_rank:", mean_rank)
					print("hits_at_10:", hits_at_10)

					print("MANY-TO-ONE")
					test_model["LINK_PREDICTION"]["MANY-TO-ONE"] = {}
					print("REPLACING HEAD ENTITY")
					test_model["LINK_PREDICTION"]["MANY-TO-ONE"]["REPLACE_HEAD"] = {}
					ranks, mean_rank, hits_at_10 = test_perform_link_prediction(dataset.many_to_one_data, test_replace_head)
					test_model["LINK_PREDICTION"]["MANY-TO-ONE"]["REPLACE_HEAD"]["ranks"] = ranks
					test_model["LINK_PREDICTION"]["MANY-TO-ONE"]["REPLACE_HEAD"]["mean_rank"] = mean_rank
					test_model["LINK_PREDICTION"]["MANY-TO-ONE"]["REPLACE_HEAD"]["hits_at_10"] = hits_at_10
					print("mean_rank:", mean_rank)
					print("hits_at_10:", hits_at_10)
					print("REPLACING TAIL ENTITY")
					test_model["LINK_PREDICTION"]["MANY-TO-ONE"]["REPLACE_TAIL"] = {}
					ranks, mean_rank, hits_at_10 = test_perform_link_prediction(dataset.many_to_one_data, test_replace_tail)
					test_model["LINK_PREDICTION"]["MANY-TO-ONE"]["REPLACE_TAIL"]["ranks"] = ranks
					test_model["LINK_PREDICTION"]["MANY-TO-ONE"]["REPLACE_TAIL"]["mean_rank"] = mean_rank
					test_model["LINK_PREDICTION"]["MANY-TO-ONE"]["REPLACE_TAIL"]["hits_at_10"] = hits_at_10
					print("mean_rank:", mean_rank)
					print("hits_at_10:", hits_at_10)

					print("MANY-TO-MANY")
					test_model["LINK_PREDICTION"]["MANY-TO-MANY"] = {}
					print("REPLACING HEAD ENTITY")
					test_model["LINK_PREDICTION"]["MANY-TO-MANY"]["REPLACE_HEAD"] = {}
					ranks, mean_rank, hits_at_10 = test_perform_link_prediction(dataset.many_to_many_data, test_replace_head)
					test_model["LINK_PREDICTION"]["MANY-TO-MANY"]["REPLACE_HEAD"]["ranks"] = ranks
					test_model["LINK_PREDICTION"]["MANY-TO-MANY"]["REPLACE_HEAD"]["mean_rank"] = mean_rank
					test_model["LINK_PREDICTION"]["MANY-TO-MANY"]["REPLACE_HEAD"]["hits_at_10"] = hits_at_10
					print("mean_rank:", mean_rank)
					print("hits_at_10:", hits_at_10)
					print("REPLACING TAIL ENTITY")
					test_model["LINK_PREDICTION"]["MANY-TO-MANY"]["REPLACE_TAIL"] = {}
					ranks, mean_rank, hits_at_10 = test_perform_link_prediction(dataset.many_to_many_data, test_replace_tail)
					test_model["LINK_PREDICTION"]["MANY-TO-MANY"]["REPLACE_TAIL"]["ranks"] = ranks
					test_model["LINK_PREDICTION"]["MANY-TO-MANY"]["REPLACE_TAIL"]["mean_rank"] = mean_rank
					test_model["LINK_PREDICTION"]["MANY-TO-MANY"]["REPLACE_TAIL"]["hits_at_10"] = hits_at_10
					print("mean_rank:", mean_rank)
					print("hits_at_10:", hits_at_10)

			if not args.no_triplet_classification:
				print("TRIPLET CLASSIFICATION")

				test_model["TRIPLET_CLASSIFICATION"] = {}

				positive_scores = []
				negative_scores = []

				for _ in tqdm(range(args.triplet_classification_times)):
					positive_triplet_classification_generator = dataset.create_positive_generator_for_triplet_classification(batch_size = args.batch_size).__iter__()
					negative_triplet_classification_generator = dataset.create_negative_generator_for_triplet_classification(batch_size = args.batch_size).__iter__()

					for __ in range(ceil(len(dataset.test_data) / args.batch_size)):
						positive_heads, positive_tails, positive_relations = positive_triplet_classification_generator.__next__()
						negative_heads, negative_tails, negative_relations = negative_triplet_classification_generator.__next__()
						
						positive_scores += embedding_model.test_model(session, positive_heads, positive_tails, positive_relations).tolist()
						negative_scores += embedding_model.test_model(session, negative_heads, negative_tails, negative_relations).tolist()

				positive_scores = numpy.array(positive_scores)
				negative_scores = numpy.array(negative_scores)

				best_accuracy = 0 

				for score in positive_scores.tolist() + negative_scores.tolist():
					accuracy = (((len(numpy.argwhere(positive_scores <= score)) / len(positive_scores)) + (len(numpy.argwhere(negative_scores > score)) / len(negative_scores))) / 2)
					
					if best_accuracy < accuracy:
						best_accuracy = accuracy
						best_score = score

				test_model["TRIPLET_CLASSIFICATION"]["positive_scores"] = positive_scores
				test_model["TRIPLET_CLASSIFICATION"]["negative_scores"] = negative_scores
				test_model["TRIPLET_CLASSIFICATION"]["best_accuracy"] = best_accuracy
				test_model["TRIPLET_CLASSIFICATION"]["best_score"] = best_score

				print("best_accuracy:", best_accuracy)
				print("best_score:", best_score)

			if args.output_file:
				try:
					model = pickle.load(open(args.output_file, "rb"))
				except:
					model = {}
				model["TEST"] = test_model
				pickle.dump(model, open(args.output_file, "wb"))
			