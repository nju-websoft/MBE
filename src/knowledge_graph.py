import collections
import os
import pickle
import logging
import copy
import torch.nn as nn
from src.data_utils import load_index
from src.data_utils import NO_OP_ENTITY_ID, NO_OP_RELATION_ID
from src.data_utils import DUMMY_ENTITY_ID, DUMMY_RELATION_ID
from src.data_utils import START_RELATION_ID
import src.utils.ops as ops
from src.utils.ops import int_var_cuda, var_cuda
from GCN.ARGCN_layer_attn import *
from collections import defaultdict as ddict


class KnowledgeGraph(nn.Module):
    """
    Each discrete knowledge graph is stored with an adjacency list.
    """
    def __init__(self, args):
        super(KnowledgeGraph, self).__init__()
        self.entity2id, self.id2entity = {}, {}
        self.relation2id, self.id2relation = {}, {}

        self.adj_list = None
        self.bandwidth = args.bandwidth
        self.args = args

        self.action_space = None
        self.action_space_buckets = None
        self.unique_r_space = None

        self.train_subjects_dict = dict()
        self.train_objects_dict = dict()
        self.dev_subjects_dict = dict()
        self.dev_objects_dict = dict()
        self.all_subjects_dict = dict()
        self.all_objects_dict = dict()

        self.train_subjects = None
        self.train_objects = None
        self.dev_subjects = None
        self.dev_objects = None
        self.all_subjects = None
        self.all_objects = None

        self.train_subject_vectors_dict = dict()
        self.train_object_vectors_dict = dict()
        self.dev_subject_vectors_dict = dict()
        self.dev_object_vectors_dict = dict()
        self.all_subject_vectors_dict = dict()
        self.all_object_vectors_dict = dict()

        self.train_subject_vectors = None
        self.train_object_vectors = None
        self.dev_subject_vectors = None
        self.dev_object_vectors = None
        self.all_subject_vectors = None
        self.all_object_vectors = None

        logging.info('** Create {} knowledge graph **'.format(args.model))
        self.load_graph_data(args.data_dir)
        self.load_all_answers(args.data_dir)
        self.load_all_answers_new_batch(args.data_dir)

        # Define NN Modules
        self.entity_dim = args.emb_dim
        self.relation_dim = args.emb_dim
        self.emb_dropout_rate = args.emb_dropout_rate
        self.entity_embeddings = None
        self.relation_embeddings = None
        self.entity_img_embeddings = None
        self.relation_img_embeddings = None
        self.EDropout = None
        self.RDropout = None

        self.data_box = {'all':dict()}
        for i in range(self.args.batch_num+1):
            self.data_box[str(i)] = dict()
        self.fill_data_box()
        if not self.args.argcn:
            self.define_modules()
            self.initialize_modules()
        else:
            self.encoder = ARGCN(self.args, self.data_box)
            self.encoder.cuda()
            self.initial_encoder(self.encoder)

    def initial_encoder(self, encoder):
        self.RDropout = nn.Dropout(self.emb_dropout_rate)
        self.EDropout = nn.Dropout(self.emb_dropout_rate)
        for parameter in encoder.parameters():
            if len(parameter.shape) == 1:
                nn.Parameter(nn.init.xavier_uniform_(parameter.unsqueeze(0)).squeeze(0))

            else:
                nn.init.xavier_uniform_(parameter)
                parameter = var_cuda(parameter, True)

    def load_graph_data(self, data_dir):
        # Load indices
        self.entity2id, self.id2entity = load_index(os.path.join(data_dir, 'entity2id.txt'))
        # logging.info('Sanity check: {} entities loaded'.format(len(self.entity2id)))
        self.relation2id, self.id2relation = load_index(os.path.join(data_dir, 'relation2id.txt'))
        # logging.info('Sanity check: {} relations loaded'.format(len(self.relation2id)))
       
        # Load graph structures
        if self.args.model.startswith('point'):
            self.vectorize_action_space(data_dir)

    def fill_data_box(self):
        '''
        We use a data box to store multi-batch data.
        data_box['0']: train
        data_box['1']: valid
        data_box['2~6']: test
        '''
        # load training data
        self.args.now_batch = '0'
        self.edge_index, self.edge_type, self.triples = [[], []], [], []
        self.data_box['0']['num_ent'] = len(self.entity2id.keys())
        self.data_box['0']['num_rel'] = len(self.relation2id.keys())
        self.load_edge_index_type(base=True)
        self.data_box['0']['edge_index'], self.data_box['0']['edge_type'] = torch.LongTensor(
            [self.edge_index[0]+self.edge_index[1], self.edge_index[1]+self.edge_index[0]]).cuda(), torch.cat([torch.LongTensor(self.edge_type).cuda(), torch.LongTensor(self.edge_type).cuda()+1], dim=0)
        self.data_box['0']['s2r'], self.data_box['0']['sr2o'], self.data_box['0']['r2s'] = self.load_s2r_sr2o_base()
        self.data_box['0']['s2r_test'], self.data_box['0']['sr2o_test'], self.data_box['0']['r2s_test'] = self.data_box['0']['s2r'], self.data_box['0']['sr2o'], self.data_box['0']['r2s']
        self.data_box['0']['s2r_support'], self.data_box['0']['sr2o_support'], self.data_box['0']['r2s_support'] = self.data_box['0']['s2r_test'], self.data_box['0']['sr2o_test'], self.data_box['0']['r2s_test']
        self.data_box['0']['edge_index_aug_link'] = self.data_box['0']['edge_index']
        self.data_box['0']['edge_type_aug_link'] = self.data_box['0']['edge_type']
        # load valid and test data
        for i in range(self.args.batch_num):
            batch_id = str(i+1)
            last_id = str(i)
            self.args.now_batch = batch_id
            self.data_box[batch_id]['num_ent'] = len(self.entity2id.keys())
            self.data_box[batch_id]['num_rel'] = len(self.relation2id.keys())
            self.load_edge_index_type()
            self.data_box[batch_id]['edge_index'], self.data_box[batch_id]['edge_type'] = torch.LongTensor(
                [self.edge_index[0]+self.edge_index[1], self.edge_index[1]+self.edge_index[0]]).cuda(), torch.cat([torch.LongTensor(self.edge_type).cuda(), torch.LongTensor(self.edge_type).cuda()+1], dim=0)
            a, b, c = copy.deepcopy(self.data_box[last_id]['s2r_support']),copy.deepcopy(self.data_box[last_id]['sr2o_support']), copy.deepcopy(self.data_box[last_id]['r2s_support'])
            a_, b_, c_ = copy.deepcopy(a), copy.deepcopy(b), copy.deepcopy(c)
            self.data_box[batch_id]['s2r_support'], self.data_box[batch_id]['sr2o_support'], self.data_box[batch_id][
                'r2s_support'], self.data_box[batch_id]['s2r_test'], self.data_box[batch_id]['sr2o_test'], \
            self.data_box[batch_id]['r2s_test'] = self.load_s2r_sr2o_new_batch(a, b, c, a_, b_, c_)
            if self.args.aug_link:
                self.data_box[batch_id]['edge_index_aug_link'] = self.data_box[batch_id]['edge_index']
                self.data_box[batch_id]['edge_type_aug_link'] = self.data_box[batch_id]['edge_type']
        self.data_box['all'] = self.data_box[str(self.args.batch_num)]
        self.args.now_batch = '0'

        self.args.num_rel = self.data_box['all']['num_rel']

    def add_aug_links(self, paths, id2path, path2id, mode='train'):
        if mode == 'train':
            s2r, sr2o, r2s = self.data_box[self.args.now_batch]['s2r'], self.data_box[self.args.now_batch]['sr2o'], self.data_box[self.args.now_batch]['r2s']
        elif mode == 'valid':
            s2r, sr2o, r2s = self.data_box[self.args.now_batch]['s2r_valid'], self.data_box[self.args.now_batch]['sr2o_valid'], self.data_box[self.args.now_batch]['r2s_valid']
        elif mode == 'test':
            s2r, sr2o, r2s = self.data_box[self.args.now_batch]['s2r_test'], self.data_box[self.args.now_batch]['sr2o_test'], self.data_box[self.args.now_batch]['r2s_test']
        def search_object(root, path, s2r, sr2o, head_r):
            now = {root}
            next = set()
            for depth in range(len(path)):
                r = path[depth]
                for s in now:
                    if s == root and r == head_r:  # make sure no rx --> rx
                        continue
                    if r in s2r[s]:
                        next = next | sr2o[(s, r)]
                now = copy.deepcopy(next)
                next.clear()
            return now
        triples = set()
        triples_info = dict()
        path_num = 0
        for head_r, body_paths in paths.items():
            if head_r == 'candidate_path' or len(body_paths) == 0:
                continue
            path_num += len(body_paths)
            candidate_s = r2s[head_r]
            for s in candidate_s:
                for p, val in body_paths.items():
                    conf, pos = val['conf'], val['pos']
                    new_o = search_object(s, p, s2r, sr2o, head_r)
                    for o in new_o:
                        if (s, head_r, o) not in triples or conf > triples_info[(s, head_r, o)]['conf']:
                            triples_info[(s, head_r, o)] = {'conf': conf, 'path': path2id[p]}
                        triples.add((s, head_r, o))
        if len(triples_info)==0:
            return None
        return triples_info

    def load_edge_index_type(self, base=False):
        if base:
            file_name = 'base_train.triples'
        else:
            if self.args.now_batch == '1':
                return
            file_name = 'add_' + self.args.now_batch + '/support.triples'
        with open(os.path.join(self.args.data_dir, file_name)) as f:
            for line in f:
                e1, e2, r = line.strip().split()
                e1, e2, r = self.triple2ids((e1, e2, r))
                self.edge_index[0].append(e1)
                self.edge_index[1].append(e2)
                self.edge_type.append(r)
                self.triples.append((e1, r, e2))


    def load_s2r_sr2o_new_batch(self, s2r_support_, sr2o_support_, r2s_support_, s2r_test_, sr2o_test_, r2s_test_):
        s2r_support, sr2o_support, r2s_support, s2r_test, sr2o_test, r2s_test = s2r_support_, sr2o_support_, r2s_support_, s2r_test_, sr2o_test_, r2s_test_
        for file_name in ['support.triples', 'test.triples']:
            if self.args.now_batch == '1':
                if file_name == 'support.triples':
                    continue
                elif file_name == 'test.triples':
                    file_name = 'valid.triples'
            with open(os.path.join(self.args.data_dir+'/add_'+self.args.now_batch, file_name)) as f:
                for line in f:
                    e1, e2, r = line.strip().split()
                    e1, e2, r = self.triple2ids((e1, e2, r))
                    if file_name == 'support.triples':
                        # get s2r
                        if e1 not in s2r_support.keys():
                            s2r_support[e1] = set()
                        if e2 not in s2r_support.keys():
                            s2r_support[e2] = set()
                        s2r_support[e1].add(r)
                        s2r_support[e2].add(self.get_inv_relation_id(r))

                        # get sr2o
                        if (e1, r) not in sr2o_support.keys():
                            sr2o_support[(e1, r)] = set()
                        if (e2, self.get_inv_relation_id(r)) not in sr2o_support.keys():
                            sr2o_support[(e2, self.get_inv_relation_id(r))] = set()
                        sr2o_support[(e1, r)].add(e2)
                        sr2o_support[(e2, self.get_inv_relation_id(r))].add(e1)

                        # get r2s
                        if r not in r2s_support.keys():
                            r2s_support[r] = set()
                        r2s_support[r].add(e1)

                    # get s2r
                    if e1 not in s2r_test.keys():
                        s2r_test[e1] = set()
                    if e2 not in s2r_test.keys():
                        s2r_test[e2] = set()
                    s2r_test[e1].add(r)
                    s2r_test[e2].add(self.get_inv_relation_id(r))

                    # get sr2o
                    if (e1, r) not in sr2o_test.keys():
                        sr2o_test[(e1, r)] = set()
                    if (e2, self.get_inv_relation_id(r)) not in sr2o_test.keys():
                        sr2o_test[(e2, self.get_inv_relation_id(r))] = set()
                    sr2o_test[(e1, r)].add(e2)
                    sr2o_test[(e2, self.get_inv_relation_id(r))].add(e1)

                    # get r2s
                    if r not in r2s_test.keys():
                        r2s_test[r] = set()
                    r2s_test[r].add(e1)
        return s2r_support, sr2o_support, r2s_support, s2r_test, sr2o_test, r2s_test

    def load_s2r_sr2o_base(self):
        s2r_test, sr2o_test, r2s_test =  dict(), dict(), dict()
        for file_name in ['base_train.triples']:
            with open(os.path.join(self.args.data_dir, file_name)) as f:
                for line in f:
                    e1, e2, r = line.strip().split()
                    e1, e2, r = self.triple2ids((e1, e2, r))
                    # get s2r
                    if e1 not in s2r_test.keys():
                        s2r_test[e1] = set()
                    if e2 not in s2r_test.keys():
                        s2r_test[e2] = set()
                    s2r_test[e1].add(r)
                    s2r_test[e2].add(self.get_inv_relation_id(r))

                    # get sr2o
                    if (e1, r) not in sr2o_test.keys():
                        sr2o_test[(e1, r)] = set()
                    if (e2, self.get_inv_relation_id(r)) not in sr2o_test.keys():
                        sr2o_test[(e2, self.get_inv_relation_id(r))] = set()
                    sr2o_test[(e1, r)].add(e2)
                    sr2o_test[(e2, self.get_inv_relation_id(r))].add(e1)

                    # get r2s
                    if r not in r2s_test.keys():
                        r2s_test[r] = set()
                    r2s_test[r].add(e1)
        return s2r_test, sr2o_test, r2s_test

    def vectorize_action_space(self, data_dir):
        """
        Pre-process and numericalize the knowledge graph structure.
        """
        if int(self.args.now_batch) > 0:
            data_dir = data_dir + '/add_' + self.args.now_batch+'/'
        adj_list_path = os.path.join(data_dir, 'adj_list.pkl')
        with open(adj_list_path, 'rb') as f:
            self.adj_list = pickle.load(f)
        def load_page_rank_scores(input_path):
            pgrk_scores = collections.defaultdict(float)
            with open(input_path) as f:
                for line in f:
                    e, score = line.strip().split(':')
                    e_id = self.entity2id[e.strip()]
                    score = float(score)
                    pgrk_scores[e_id] = score
            return pgrk_scores
                    
        # Sanity check
        num_facts = 0
        out_degrees = collections.defaultdict(int)
        for e1 in self.adj_list:
            for r in self.adj_list[e1]:
                num_facts += len(self.adj_list[e1][r])
                out_degrees[e1] += len(self.adj_list[e1][r])
        # logging.info("Sanity check: maximum out degree: {}".format(max(out_degrees.values())))
        # logging.info('Sanity check: {} facts in knowledge graph'.format(num_facts))

        # load page rank scores
        page_rank_scores = load_page_rank_scores(os.path.join(data_dir, 'node.pgrk'))
        self.page_rank_scores = page_rank_scores

        def get_action_space(e1):
            action_space = []
            if e1 in self.adj_list:
                for r in self.adj_list[e1]:
                    targets = self.adj_list[e1][r]
                    for e2 in targets:
                        action_space.append((r, e2, 1.0, 0))
                if len(action_space) + 1 >= self.bandwidth:
                    # Base graph pruning
                    sorted_action_space = \
                        sorted(action_space, key=lambda x: page_rank_scores[x[1]], reverse=True)
                    action_space = sorted_action_space[:self.bandwidth]
            action_space.insert(0, (NO_OP_RELATION_ID, e1, 1.0, 0))
            return action_space

        def get_unique_r_space(e1):
            if e1 in self.adj_list:
                return list(self.adj_list[e1].keys())
            else:
                return []

        def vectorize_action_space(action_space_list, action_space_size):
            bucket_size = len(action_space_list)
            r_space = torch.zeros(bucket_size, action_space_size) + self.dummy_r
            e_space = torch.zeros(bucket_size, action_space_size) + self.dummy_e
            c_space = torch.zeros(bucket_size, action_space_size)
            p_space = torch.zeros(bucket_size, action_space_size)
            action_mask = torch.zeros(bucket_size, action_space_size)
            for i, action_space in enumerate(action_space_list):
                for j, (r, e, c, p) in enumerate(action_space):
                    r_space[i, j] = r
                    e_space[i, j] = e
                    c_space[i, j] = c
                    p_space[i, j] = p
                    action_mask[i, j] = 1
            return (int_var_cuda(r_space), int_var_cuda(e_space), var_cuda(c_space), int_var_cuda(p_space)), var_cuda(action_mask)

        def vectorize_unique_r_space(unique_r_space_list, unique_r_space_size, volatile):
            bucket_size = len(unique_r_space_list)
            unique_r_space = torch.zeros(bucket_size, unique_r_space_size) + self.dummy_r
            for i, u_r_s in enumerate(unique_r_space_list):
                for j, r in enumerate(u_r_s):
                    unique_r_space[i, j] = r
            return int_var_cuda(unique_r_space)

        if self.args.use_action_space_bucketing:
            """
            Store action spaces in buckets.
            """
            self.action_space_buckets = {}
            action_space_buckets_discrete = collections.defaultdict(list)
            self.entity2bucketid = torch.zeros(self.num_entities, 2).long()
            num_facts_saved_in_action_table = 0
            for e1 in range(self.num_entities):
                action_space = get_action_space(e1)
                key = int(len(action_space) / self.args.bucket_interval) + 1
                self.entity2bucketid[e1, 0] = key
                self.entity2bucketid[e1, 1] = len(action_space_buckets_discrete[key])
                action_space_buckets_discrete[key].append(action_space)
                num_facts_saved_in_action_table += len(action_space)
            for key in action_space_buckets_discrete:
                self.action_space_buckets[key] = vectorize_action_space(
                    action_space_buckets_discrete[key], key * self.args.bucket_interval)
        else:
            action_space_list = []
            max_num_actions = 0
            for e1 in range(self.num_entities):
                action_space = get_action_space(e1)
                action_space_list.append(action_space)
                if len(action_space) > max_num_actions:
                    max_num_actions = len(action_space)
            logging.info('Vectorizing action spaces...')
            self.action_space = vectorize_action_space(action_space_list, max_num_actions)
            
            if self.args.model.startswith('rule'):
                unique_r_space_list = []
                max_num_unique_rs = 0
                for e1 in sorted(self.adj_list.keys()):
                    unique_r_space = get_unique_r_space(e1)
                    unique_r_space_list.append(unique_r_space)
                    if len(unique_r_space) > max_num_unique_rs:
                        max_num_unique_rs = len(unique_r_space)
                self.unique_r_space = vectorize_unique_r_space(unique_r_space_list, max_num_unique_rs)

    def vectorize_action_space_aug_link(self, aug_links):
        """
        Pre-process and numericalize the knowledge graph structure.

        """
        if int(self.args.now_batch) > 0:
            data_dir = self.args.data_dir + '/add_' + self.args.now_batch+'/'
        else:
            data_dir = self.args.data_dir
        adj_list_path = os.path.join(data_dir, 'adj_list.pkl')
        with open(adj_list_path, 'rb') as f:
            self.adj_list = pickle.load(f)

        self.adj_list_aug = dict()

        def expand_adj_list(aug_links):
            edge_conf = ddict(list)
            if aug_links == None:
                return edge_conf
            for triple, val in aug_links.items():
                h, r, t = triple
                if h not in self.adj_list_aug.keys():
                    self.adj_list_aug[h] = ddict(list)
                self.adj_list_aug[h][r].append(t)
                edge_conf[triple].append([val['conf'], val['path']])
            return edge_conf

        # load page rank scores
        page_rank_scores = self.page_rank_scores

        def get_action_space_aug(e1, conf):
            action_space = []
            if e1 in self.adj_list:
                for r in self.adj_list[e1]:
                    targets = self.adj_list[e1][r]
                    for e2 in targets:
                        action_space.append((r, e2, 1.0, 0))
            if e1 in self.adj_list_aug:
                for r in self.adj_list_aug[e1]:
                    targets = self.adj_list_aug[e1][r]
                    for e2 in targets:
                        for c, p in conf[(e1, r, e2)]:
                            action_space.append((r, e2, c, p))
            if len(action_space) + 1 >= self.bandwidth:
                # Base graph pruning
                sorted_action_space = \
                    sorted(action_space, key=lambda x: page_rank_scores[x[1]], reverse=True)
                action_space = sorted_action_space[:self.bandwidth]
            action_space.insert(0, (NO_OP_RELATION_ID, e1, 1.0, 0))
            return action_space

        def vectorize_action_space(action_space_list, action_space_size):
            bucket_size = len(action_space_list)
            r_space = torch.zeros(bucket_size, action_space_size) + self.dummy_r
            e_space = torch.zeros(bucket_size, action_space_size) + self.dummy_e
            c_space = torch.zeros(bucket_size, action_space_size)
            p_space = torch.zeros(bucket_size, action_space_size)
            action_mask = torch.zeros(bucket_size, action_space_size)
            for i, action_space in enumerate(action_space_list):
                for j, (r, e, c, p) in enumerate(action_space):
                    r_space[i, j] = r
                    e_space[i, j] = e
                    c_space[i, j] = c
                    action_mask[i, j] = 1
                    p_space[i, j] = p
            return (int_var_cuda(r_space), int_var_cuda(e_space), var_cuda(c_space), int_var_cuda(p_space)), var_cuda(action_mask)

        edge_conf = expand_adj_list(aug_links)
        """
        Store action spaces in buckets.
        """
        self.action_space_buckets = {}
        action_space_buckets_discrete = collections.defaultdict(list)
        self.entity2bucketid = torch.zeros(self.num_entities, 2).long()
        num_facts_saved_in_action_table = 0
        for e1 in range(self.num_entities):
            action_space = get_action_space_aug(e1, edge_conf)
            key = int(len(action_space) / self.args.bucket_interval) + 1
            self.entity2bucketid[e1, 0] = key
            self.entity2bucketid[e1, 1] = len(action_space_buckets_discrete[key])
            action_space_buckets_discrete[key].append(action_space)
            num_facts_saved_in_action_table += len(action_space)

        for key in action_space_buckets_discrete:
            self.action_space_buckets[key] = vectorize_action_space(
                action_space_buckets_discrete[key], key * self.args.bucket_interval)

    def load_all_answers(self, data_dir, add_reversed_edges=False):
        def add_subject(e1, e2, r, d):
            if not e2 in d:
                d[e2] = {}
            if not r in d[e2]:
                d[e2][r] = set()
            d[e2][r].add(e1)

        def add_object(e1, e2, r, d):
            if not e1 in d:
                d[e1] = {}
            if not r in d[e1]:
                d[e1][r] = set()
            d[e1][r].add(e2)

        # store subjects for all (rel, object) queries and objects for all (subject, rel) queries
        train_subjects, train_objects = {}, {}
        dev_subjects, dev_objects = {}, {}
        all_subjects, all_objects = {}, {}
        # include dummy examples
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, train_subjects)
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, dev_subjects)
        add_subject(self.dummy_e, self.dummy_e, self.dummy_r, all_subjects)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, train_objects)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, dev_objects)
        add_object(self.dummy_e, self.dummy_e, self.dummy_r, all_objects)

        file_name = 'base_train.triples'
        with open(os.path.join(data_dir, file_name)) as f:
            for line in f:
                e1, e2, r = line.strip().split()
                e1, e2, r = self.triple2ids((e1, e2, r))
                add_subject(e1, e2, r, train_subjects)
                add_object(e1, e2, r, train_objects)
                if add_reversed_edges:
                    add_subject(e2, e1, self.get_inv_relation_id(r), train_subjects)
                    add_object(e2, e1, self.get_inv_relation_id(r), train_objects)
                add_subject(e1, e2, r, dev_subjects)
                add_object(e1, e2, r, dev_objects)
                if add_reversed_edges:
                    add_subject(e2, e1, self.get_inv_relation_id(r), dev_subjects)
                    add_object(e2, e1, self.get_inv_relation_id(r), dev_objects)
                add_subject(e1, e2, r, all_subjects)
                add_object(e1, e2, r, all_objects)
                if add_reversed_edges:
                    add_subject(e2, e1, self.get_inv_relation_id(r), all_subjects)
                    add_object(e2, e1, self.get_inv_relation_id(r), all_objects)
        self.train_subjects_dict['0'] = train_subjects
        self.train_objects_dict['0'] = train_objects
        self.dev_subjects_dict['0'] = dev_subjects
        self.dev_objects_dict['0'] = dev_objects
        self.all_subjects_dict['0'] = all_subjects
        self.all_objects_dict['0'] = all_objects

        # change the answer set into a variable
        def answers_to_var(d_l):
            d_v = collections.defaultdict(collections.defaultdict)
            for x in d_l:
                for y in d_l[x]:
                    v = torch.LongTensor(list(d_l[x][y])).unsqueeze(1)
                    d_v[x][y] = int_var_cuda(v)
            return d_v

        self.train_subject_vectors_dict['0'] = answers_to_var(train_subjects)
        self.train_object_vectors_dict['0'] = answers_to_var(train_objects)
        self.dev_subject_vectors_dict['0'] = answers_to_var(dev_subjects)
        self.dev_object_vectors_dict['0'] = answers_to_var(dev_objects)
        self.all_subject_vectors_dict['0'] = answers_to_var(all_subjects)
        self.all_object_vectors_dict['0'] = answers_to_var(all_objects)

    def load_all_answers_new_batch(self, data_dir, add_reversed_edges=False):
        def add_subject(e1, e2, r, d):
            if not e2 in d:
                d[e2] = {}
            if not r in d[e2]:
                d[e2][r] = set()
            d[e2][r].add(e1)

        def add_object(e1, e2, r, d):
            if not e1 in d:
                d[e1] = {}
            if not r in d[e1]:
                d[e1][r] = set()
            d[e1][r].add(e2)

        # change the answer set into a variable
        def answers_to_var(d_l):
            d_v = collections.defaultdict(collections.defaultdict)
            for x in d_l:
                for y in d_l[x]:
                    v = torch.LongTensor(list(d_l[x][y])).unsqueeze(1)
                    d_v[x][y] = int_var_cuda(v)
            return d_v

        def merge_dict(dict1, dict2):
            for e1 in dict2.keys():
                if e1 not in dict1.keys():
                    dict1[e1] = dict2[e1]
                    continue
                for r in dict2[e1].keys():
                    if r not in dict1[e1].keys():
                        dict1[e1][r] = dict2[e1][r]
                    else:
                        dict1[e1][r] = dict2[e1][r] | dict1[e1][r]
            return dict1

        # store subjects for all (rel, object) queries and
        # objects for all (subject, rel) queries
        for i in range(self.args.batch_num):
            batch_id = str(i+1)
            train_subjects, train_objects = {}, {}
            dev_subjects, dev_objects = {}, {}
            all_subjects, all_objects = {}, {}
            # include dummy examples
            add_subject(self.dummy_e, self.dummy_e, self.dummy_r, train_subjects)
            add_subject(self.dummy_e, self.dummy_e, self.dummy_r, dev_subjects)
            add_subject(self.dummy_e, self.dummy_e, self.dummy_r, all_subjects)
            add_object(self.dummy_e, self.dummy_e, self.dummy_r, train_objects)
            add_object(self.dummy_e, self.dummy_e, self.dummy_r, dev_objects)
            add_object(self.dummy_e, self.dummy_e, self.dummy_r, all_objects)
            for file_name in ['support.triples', 'test.triples']:
                if batch_id == '1':
                    if file_name == 'support.triples':
                        continue
                    elif file_name == 'test.triples':
                        file_name = 'valid.triples'
                with open(os.path.join(data_dir+'/add_'+batch_id, file_name)) as f:
                    for line in f:
                        e1, e2, r = line.strip().split()
                        e1, e2, r = self.triple2ids((e1, e2, r))
                        if file_name in ['support.triples']:
                            add_subject(e1, e2, r, train_subjects)
                            add_object(e1, e2, r, train_objects)
                            if add_reversed_edges:
                                add_subject(e2, e1, self.get_inv_relation_id(r), train_subjects)
                                add_object(e2, e1, self.get_inv_relation_id(r), train_objects)
                            add_subject(e1, e2, r, dev_subjects)
                            add_object(e1, e2, r, dev_objects)
                            if add_reversed_edges:
                                add_subject(e2, e1, self.get_inv_relation_id(r), dev_subjects)
                                add_object(e2, e1, self.get_inv_relation_id(r), dev_objects)
                        add_subject(e1, e2, r, all_subjects)
                        add_object(e1, e2, r, all_objects)
                        if add_reversed_edges:
                            add_subject(e2, e1, self.get_inv_relation_id(r), all_subjects)
                            add_object(e2, e1, self.get_inv_relation_id(r), all_objects)
            last_batch_id = str(int(batch_id)-1)


            self.train_subjects_dict[batch_id] = merge_dict(train_subjects, self.train_subjects_dict[last_batch_id])
            self.train_objects_dict[batch_id] = merge_dict(train_objects, self.train_objects_dict[last_batch_id])
            self.dev_subjects_dict[batch_id] = merge_dict(dev_subjects, self.dev_subjects_dict[last_batch_id])
            self.dev_objects_dict[batch_id] = merge_dict(dev_objects, self.dev_objects_dict[last_batch_id])
            self.all_subjects_dict[batch_id] = merge_dict(all_subjects, self.all_subjects_dict[last_batch_id])
            self.all_objects_dict[batch_id] = merge_dict(all_objects, self.all_objects_dict[last_batch_id])

            self.train_subject_vectors_dict[batch_id] = answers_to_var(self.train_subjects_dict[batch_id])
            self.train_object_vectors_dict[batch_id] = answers_to_var(self.train_objects_dict[batch_id])
            self.dev_subject_vectors_dict[batch_id] = answers_to_var(self.dev_subjects_dict[batch_id])
            self.dev_object_vectors_dict[batch_id] = answers_to_var(self.dev_objects_dict[batch_id])
            self.all_subject_vectors_dict[batch_id] = answers_to_var(self.all_subjects_dict[batch_id])
            self.all_object_vectors_dict[batch_id] = answers_to_var(self.all_objects_dict[batch_id])

    def prepare_new_batch(self, batch_id):
        self.args.now_batch = batch_id
        if self.args.argcn:
            self.update_embedding(batch_id)
        self.vectorize_action_space(self.args.data_dir)
        self.train_subjects = self.train_subjects_dict[batch_id]
        self.train_objects = self.train_objects_dict[batch_id]
        self.dev_subjects = self.dev_subjects_dict[batch_id]
        self.dev_objects = self.dev_objects_dict[batch_id]
        self.all_subjects = self.all_subjects_dict[batch_id]
        self.all_objects = self.all_objects_dict[batch_id]

        self.train_subject_vectors = self.train_subject_vectors_dict[batch_id]
        self.train_object_vectors = self.train_object_vectors_dict[batch_id]
        self.dev_subject_vectors = self.dev_subject_vectors_dict[batch_id]
        self.dev_object_vectors = self.dev_object_vectors_dict[batch_id]
        self.all_subject = self.all_subject_vectors_dict[batch_id]
        self.all_object_vectors = self.all_object_vectors_dict[batch_id]


    def get_inv_relation_id(self, r_id):
        return r_id + 1

    def get_all_entity_embeddings(self):
        if self.args.argcn:
            return self.EDropout(self.entity_embeddings)
        else:
            return self.EDropout(self.entity_embeddings.weight)

    def get_entity_embeddings(self, e):
        if self.args.argcn:
            return self.EDropout(self.entity_embeddings[e])
        else:
            return self.EDropout(self.entity_embeddings.weight[e])

    def get_all_relation_embeddings(self):
        if self.args.argcn:
            return self.RDropout(self.relation_embeddings)
        else:
            return self.RDropout(self.relation_embeddings.weight)

    def get_relation_embeddings(self, r):
        if self.args.argcn:
            return self.RDropout(self.relation_embeddings[r])
        else:
            return self.RDropout(self.relation_embeddings.weight[r])

    def id2triples(self, triple):
        e1, e2, r = triple
        return self.id2entity[e1], self.id2entity[e2], self.id2relation[r]

    def triple2ids(self, triple):
        e1, e2, r = triple
        return self.entity2id[e1], self.entity2id[e2], self.relation2id[r]

    def define_modules(self):
        self.entity_embeddings = nn.Embedding(self.num_entities, self.entity_dim)
        self.EDropout = nn.Dropout(self.emb_dropout_rate)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_dim)
        self.RDropout = nn.Dropout(self.emb_dropout_rate)

    def update_embedding(self, batch, query=None):
        self.entity_embeddings, self.relation_embeddings = self.encoder(batch, query)

    def initialize_modules(self):
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

    @property
    def num_entities(self):
        return len(self.entity2id)

    @property
    def num_relations(self):
        return len(self.relation2id)

    @property
    def self_edge(self):
        return NO_OP_RELATION_ID

    @property
    def self_e(self):
        return NO_OP_ENTITY_ID        

    @property
    def dummy_r(self):
        return DUMMY_RELATION_ID

    @property
    def dummy_e(self):
        return DUMMY_ENTITY_ID

    @property
    def dummy_start_r(self):
        return START_RELATION_ID
