import copy
import itertools
import numpy as np
import os, sys
import random
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
import prettytable as pt

from src.parse_args import parser
from src.parse_args import args
import src.data_utils as data_utils
import src.eval
from src.knowledge_graph import KnowledgeGraph
from src.rl.graph_search.pn import GraphSearchPolicy
from src.rl.graph_search.pg import PolicyGradient
from src.utils.ops import flatten
from src.config import config
import logging
import time


def process_data():
    data_dir = args.data_dir
    raw_kb_path = os.path.join(data_dir, 'base_train.triples')
    train_path = os.path.join(data_dir, 'base_train.triples')
    valid_path = [os.path.join(data_dir, 'add_1/valid.triples')]
    test_path = valid_path + [os.path.join(data_dir, 'add_' + str(i + 1) + '/test.triples') for i in range(1, args.batch_num)]
    support_path = [os.path.join(data_dir, 'add_' + str(i + 1) + '/support.triples') for i in
                    range(args.batch_num)]
    data_utils.prepare_kb_envrioment(args.batch_num, raw_kb_path, train_path, test_path, support_path,
                                               args.add_reverse_relations)


def train(lf):
    train_path = data_utils.get_train_path(args)
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    train_data = data_utils.load_triples(
        train_path, entity_index_path, relation_index_path, add_reverse_relations=args.add_reversed_training_edges)
    if args.checkpoint_path is not None:
        lf.load_checkpoint(args.checkpoint_path)
    lf.run_train(train_data)


def inference(lf):
    lf.batch_size = args.dev_batch_size
    lf.eval()
    lf.load_paths(best=True)
    lf.load_checkpoint(get_checkpoint_path(args))
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    eval_metrics = {
        'dev': {},
        'test': {}
    }
    # if args.vs100:
    #     corrupts_path = [os.path.join(args.data_dir, 'add_' + str(i + 1) + '/corrupt.txt') for i in
    #                      range(1, args.batch_num)]
    #     data_dir = args.data_dir
    #     corrupts = data_utils.read_corrupt(corrupts_path, data_dir)

    tb_1, tb_2 = pt.PrettyTable(), pt.PrettyTable()
    tb_1.field_names = ['GroupID', 'Hits@1', 'Hits@3', 'Hits@5', 'Hits@10', 'MRR']
    for i in range(1, args.batch_num):
        print('Testing Add Group:', i)
        batch_id = str(i+1)
        lf.kg.prepare_new_batch(batch_id)
        test_path = os.path.join(args.data_dir, 'add_' + batch_id + '/' + 'test.triples')
        test_data_ = data_utils.load_triples(test_path, entity_index_path, relation_index_path,
                                  seen_entities=set())
        torch.cuda.empty_cache()
        args.now_batch = batch_id
        lf.kg.prepare_new_batch(batch_id)
        if args.aug_link:
            lf.kg.aug_link_info = lf.kg.add_aug_links(lf.path_record['candidate_path'],
                                                              lf.id2path, lf.path2id,
                                                              mode='test')
            if lf.kg.aug_link_info is not None:
                lf.kg.vectorize_action_space_aug_link(lf.kg.aug_link_info)
        test_scores, test_data__ = lf.forward(test_data_, verbose=False)
        # if args.vs100:
        #     hits_1_, hits_3_, hits_5_, hits_10_, mrr_ = src.eval.hits_and_ranks_vs100(test_data__, test_scores,
        #                                                                                lf.kg.all_objects,
        #                                                                                corrupts[i],
        #                                                                                verbose=True)
        # else:
        hits_1_, hits_3_, hits_5_, hits_10_, mrr_ = src.eval.hits_and_ranks(test_data__, test_scores,
                                                                                lf.kg.all_objects, verbose=True)
        tb_1.add_row([i, round(hits_1_, 3), round(hits_3_, 3), round(hits_5_, 3), round(hits_10_, 3),
                      round(mrr_, 3)])
    logging.info(tb_1)

    return eval_metrics


def get_checkpoint_path(args):
    if not args.checkpoint_path:
        return os.path.join(args.model_dir, 'model_best.tar')
    else:
        return args.checkpoint_path


def run_experiment(args):
    if args.process_data:
        # Process knowledge graph data
        process_data()
    else:
        # Build model
        initialize_model_directory(args)
        lf = construct_model(args)
        lf.cuda()
        if args.train:
            train(lf)
        elif args.inference:
            inference(lf)


def initialize_model_directory(args):
    # add model parameter info to model directory
    model_root_dir = args.model_root_dir
    dataset = os.path.basename(os.path.normpath(args.data_dir))

    reverse_edge_tag = '-RV' if args.add_reversed_training_edges else ''
    entire_graph_tag = '-EG' if args.train_entire_graph else ''
    if args.xavier_initialization:
        initialization_tag = '-xavier'
    elif args.uniform_entity_initialization:
        initialization_tag = '-uniform'
    else:
        initialization_tag = ''

    if args.aug_link and args.argcn and args.attn:
        ablation = 'ourmodel'
    else:
        ablation = 'w/o'

    if not args.argcn:
        ablation += '_argcn_'
    if not args.aug_link:
        ablation += '_aug_link_'
    if not args.attn:
        ablation += '_attn_'
    args.ablation_path = ablation


    # Hyperparameter signature
    if args.model in ['rule']:
        hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            args.emb_dim,
            args.history_num_layers,
            args.learning_rate,
            args.emb_dropout_rate,
            args.ff_dropout_rate,
            args.action_dropout_rate,
            args.bandwidth,
            args.beta,
            ablation
        )
    elif args.model.startswith('point'):
        if args.action_dropout_anneal_interval < 1000:
            hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
                args.emb_dim,
                args.history_num_layers,
                args.learning_rate,
                args.emb_dropout_rate,
                args.ff_dropout_rate,
                args.action_dropout_rate,
                args.action_dropout_anneal_factor,
                args.action_dropout_anneal_interval,
                args.bandwidth,
                args.beta,
                ablation
            )
            if args.mu != 1.0:
                hyperparam_sig += '-{}'.format(args.mu)
        else:
            hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
                args.emb_dim,
                args.history_num_layers,
                args.learning_rate,
                args.emb_dropout_rate,
                args.ff_dropout_rate,
                args.action_dropout_rate,
                args.bandwidth,
                args.beta,
                ablation
            )
    else:
        raise NotImplementedError

    model_sub_dir = '{}-{}{}{}{}-{}-{}'.format(
        dataset,
        args.model,
        reverse_edge_tag,
        entire_graph_tag,
        initialization_tag,
        hyperparam_sig,
        ablation
    )

    model_dir = os.path.join(model_root_dir, model_sub_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print('Model directory created: {}'.format(model_dir))
    else:
        print('Model directory exists: {}'.format(model_dir))

    args.model_dir = model_dir


def set_logger():
    dataset = os.path.basename(os.path.normpath(args.data_dir))
    if args.train:
        log_filename = "../logs/" + dataset + str(time.time()) + 'train.log'
    elif args.inference:
        log_filename = "../logs/" + 'test.log'
    else:
        log_filename = "../logs/" + dataset + 'pre_progress.log'
    logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w',
                        format='%(asctime)-15s %(levelname)-8s %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("\x1b[38;20m" + ' %(message)s' + "\x1b[0m"))
    logging.getLogger().addHandler(console)
    logging.info(args)


def construct_model(args):
    """
    Construct NN graph.
    """
    kg = KnowledgeGraph(args)
    pn = GraphSearchPolicy(args)
    lf = PolicyGradient(args, kg, pn)
    return lf


if __name__ == '__main__':
    args.data_dir = args.data_dir + args.dataset
    torch.cuda.set_device(args.gpu)

    # setting
    config(args)
    set_logger()

    run_experiment(args)
