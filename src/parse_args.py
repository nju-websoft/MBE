import argparse
import os


parser = argparse.ArgumentParser(description='MBE')

# Experiment control
parser.add_argument('--process_data', action='store_true',
                    help='process knowledge graph (default: False)')
parser.add_argument('--train', action='store_true',
                    help='run path selection set_policy training (default: False)')
parser.add_argument('--inference', action='store_true',
                    help='run knowledge graph inference (default: False)')

parser.add_argument('--dataset', type=str, default='WN-MBE',
                    help='dataset (default: WN-MBE, FB-MBE, NELL-MBE)')
parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/'),
                    help='directory where the knowledge graph data is stored (default: None)')
parser.add_argument('--model_root_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model'),
                    help='root directory where the model parameters are stored (default: None)')
parser.add_argument('--model_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model'),
                    help='directory where the model parameters are stored (default: None)')
parser.add_argument('--gpu', dest='gpu', type=int, default=0,
                    help='gpu device (default: 0)')
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='path to a pretrained checkpoint')

# Network Architecture
parser.add_argument('--model', type=str, default='point',
                    help='knowledge graph QA model (default: point)')
parser.add_argument('--emb_dim', type=int, default=100, metavar='E',
                    help='embedding dimension (default: 100)')
parser.add_argument('--history_dim', type=int, default=100, metavar='H',
                    help='action history encoding LSTM hidden states dimension (default: 400)')
parser.add_argument('--history_num_layers', type=int, default=3, metavar='L',
                    help='action history encoding LSTM number of layers (default: 1)')
parser.add_argument('--use_action_space_bucketing', type=bool, default=True,
                    help='bucket adjacency list by outgoing degree to avoid memory blow-up (default: True)')
parser.add_argument('--bucket_interval', type=int, default=10,
                    help='adjacency list bucket size (default: 32)')

# Optimization
parser.add_argument('--num_epochs', type=int, default=200,
                    help='maximum number of pass over the entire training set (default: 20)')
parser.add_argument('--num_wait_epochs', type=int, default=5,
                    help='number of epochs to wait before stopping training if dev set performance drops')
parser.add_argument('--num_peek_epochs', type=int, default=2,
                    help='number of epochs to wait for next dev set result check (default: 2)')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='epoch from which the training should start (default: 0)')
parser.add_argument('--batch_size', type=int, default=256,
                    help='mini-batch size (default: 256)')
parser.add_argument('--train_batch_size', type=int, default=256,
                    help='mini-batch size during training (default: 256)')
parser.add_argument('--dev_batch_size', type=int, default=64,
                    help='mini-batch size during inferece (default: 64)')
parser.add_argument('--margin', type=float, default=0,
                    help='margin used for base MAMES training (default: 0)')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--learning_rate_decay', type=float, default=1.0,
                    help='learning rate decay factor for the Adam optimizer (default: 1)')
parser.add_argument('--adam_beta1', type=float, default=0.9,
                    help='Adam: decay rates for the first movement estimate (default: 0.9)')
parser.add_argument('--adam_beta2', type=float, default=0.999,
                    help='Adam: decay rates for the second raw movement estimate (default: 0.999)')
parser.add_argument('--grad_norm', type=float, default=10000,
                    help='norm threshold for gradient clipping (default 10000)')
parser.add_argument('--xavier_initialization', type=bool, default=True,
                    help='Initialize all model parameters using xavier initialization (default: True)')

# Policy Network
parser.add_argument('--ff_dropout_rate', type=float, default=0.1,
                    help='Feed-forward layer dropout rate (default: 0.1)')
parser.add_argument('--rnn_dropout_rate', type=float, default=0.0,
                    help='RNN Variational Dropout Rate (default: 0.0)')
parser.add_argument('--action_dropout_rate', type=float, default=0.1,
                    help='Dropout rate for randomly masking out knowledge graph edges (default: 0.1)')
parser.add_argument('--action_dropout_anneal_factor', type=float, default=0.95,
	                help='Decrease the action dropout rate once the dev set results stopped increase (default: 0.95)')
parser.add_argument('--action_dropout_anneal_interval', type=int, default=1000,
		            help='Number of epochs to wait before decreasing the action dropout rate (default: 1000. Action '
                         'dropout annealing is not used when the value is >= 1000.)')
parser.add_argument('--num_negative_samples', type=int, default=10,
                    help='Number of negative samples to use for embedding-based methods')

# Graph Completion
parser.add_argument('--theta', type=float, default=0.2,
                    help='Threshold for sifting high-confidence facts (default: 0.2)')

# Reinforcement Learning
parser.add_argument('--num_rollouts', type=int, default=20,
                    help='number of rollouts (default: 20)')
parser.add_argument('--num_rollout_steps', type=int, default=3,
                    help='maximum path length (default: 3)')
parser.add_argument('--bandwidth', type=int, default=300,
                    help='maximum number of outgoing edges to explore at each step (default: 300)')
parser.add_argument('--beta', type=float, default=0.0,
                    help='entropy regularization weight (default: 0.0)')
parser.add_argument('--gamma', type=float, default=1,
                    help='moving average weight (default: 1)')

# Search Decoding
parser.add_argument('--beam_size', type=int, default=100,
                    help='size of beam used in beam search inference (default: 100)')
'''
Note that the embedding- and rule-based baselines all mask false negative facts in the dev/test set, 
so we also set the mask_test_false negatives as 'True'
And Multi-Hop, GR, RuleGuider also use the same setting
'''
parser.add_argument('--mask_test_false_negatives', type=bool, default=True,
                    help='mask false negative examples in the dev/test set during decoding (default: True. '
                         'Use the same filter settings as other baseline methods.)')
parser.add_argument('--save_beam_search_paths', action='store_true',
                    help='save the decoded path into a CSV file (default: False)')

# MBE parameters
parser.add_argument('--batch_num', type=int, default=6,
                    help='the number of new batch and original KG (default: 5+1=6)')
parser.add_argument('--now_batch', type=int, default='0',
                    help='indicate the currently used data(train: 0, valid: 1; new batch: 2-6)')

# ablation study
parser.add_argument('--argcn', type=bool, default=True,
                    help='If true, the model will use ARGCN to generate embeddings (default: True)')
parser.add_argument('--aug_link', type=bool, default=True,
                    help='If true, the model will use augmentation links (default: True)')
parser.add_argument('--attn', type=bool, default=True,
                    help='If true, the model will use feedback attention (default: True)')

# model details
# ARGCN
parser.add_argument('--rel_agg', type=str, default='sum',
                    help='The pooling function of the relational convolutional layer (the first layer) (default: sum)')
parser.add_argument('--ent_agg', type=str, default='sum',
                    help='The aggregation method of the stacked layers (default: sum)')
parser.add_argument('--neigh_dropout', type=float, default=0.3,
                    help='Dropout rate of neighboring entities (default: 0.3)')
parser.add_argument('--node_dropout', type=float, default=0.3,
                    help='Dropout rate of entity embeddings (default: 0.3)')
parser.add_argument('--gcn_layer', type=int, default=1,
                    help='GCN layer (default: 1)')
# Augmentation link
parser.add_argument('--aug_link_threshold', type=float, default=0.3,
                    help='Confidence threshold value (default: 0.3)')
parser.add_argument('--aug_link_support_threshold', type=float, default=1.0,
                    help='Support threshold value (default: 1.0). value = predict_pos / groundtruth_pos. '
                         'Note the the rollout num = 20, so the value is in [0,20].')

# Evaluation parameters
parser.add_argument('--vs100', type=bool, default=False,
                    help='If true, the model will be evaluated with a 1vs100 setting (default: False)')
parser.add_argument('--run_analysis', action='store_true',
                    help='If true, the model will be evaluated on both validation and testing sets (default: False)')

# Knowledge Graph
parser.add_argument('--add_reverse_relations', type=bool, default=True,
                    help='add reverse relations to KB (default: True)')
parser.add_argument('--add_reversed_training_edges', action='store_true',
                    help='add reversed edges to extend training set (default: False)')
parser.add_argument('--train_entire_graph', type=bool, default=False,
                    help='add all edges in the graph to extend training set (default: False)')
parser.add_argument('--emb_dropout_rate', type=float, default=0.3,
                    help='Knowledge graph embedding dropout rate (default: 0.3)')
parser.add_argument('--zero_entity_initialization', type=bool, default=False,
                    help='Initialize all entities to zero (default: False)')
parser.add_argument('--uniform_entity_initialization', type=bool, default=False,
                    help='Initialize all entities with the same random embedding (default: False)')

args = parser.parse_args()
