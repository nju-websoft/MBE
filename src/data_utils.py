import collections
import numpy as np
import os
import pickle
import csv

START_RELATION = 'START_RELATION'
NO_OP_RELATION = 'NO_OP_RELATION'
NO_OP_ENTITY = 'NO_OP_ENTITY'
DUMMY_RELATION = 'DUMMY_RELATION'
DUMMY_ENTITY = 'DUMMY_ENTITY'

DUMMY_RELATION_ID = 0
START_RELATION_ID = 1
NO_OP_RELATION_ID = 2
DUMMY_ENTITY_ID = 0
NO_OP_ENTITY_ID = 1


def get_train_path(args):
    train_path = os.path.join(args.data_dir, 'base_train.triples')
    return train_path


def load_triples(data_path, entity_index_path, relation_index_path, group_examples_by_query=False,
                 add_reverse_relations=False, seen_entities=None, verbose=False):
    """
    Convert triples stored on disc into indices.
    """
    entity2id, _ = load_index(entity_index_path)
    relation2id, _ = load_index(relation_index_path)

    def triple2ids(e1, e2, r):
        return entity2id[e1], entity2id[e2], relation2id[r]

    triples = []
    with open(data_path) as f:
        num_skipped = 0
        for line in f:
            e1, e2, r = line.strip().split()
            if seen_entities and (not e1 in seen_entities or not e2 in seen_entities):
                num_skipped += 1
                if verbose:
                    print('Skip triple ({}) with unseen entity: {}'.format(num_skipped, line.strip())) 
                continue
            triples.append(triple2ids(e1, e2, r))
            if add_reverse_relations:
                triples.append(triple2ids(e2, e1, r + '_inv'))
    print('{} triples loaded from {}'.format(len(triples), data_path))
    return triples


def read_corrupt(corrupts_path, data_dir):
    entity2id, id2entity = load_index(os.path.join(data_dir, 'entity2id.txt'))
    relation2id, id2relation = load_index(os.path.join(data_dir, 'relation2id.txt'))
    corrupts = {}
    for i, path in enumerate(corrupts_path):
        corrupts[i+1] = {}
        with open(path,'r') as f:
            for line in f:
                line = line.replace('\n','').split(' ')
                sub, obj, rel = entity2id[line[0]], entity2id[line[1]], relation2id[line[2]]
                corrupts[i+1][(sub, obj, rel)] = []
                for ent in line[3:]:
                    corrupts[i+1][(sub, obj, rel)].append(entity2id[ent])
    return corrupts


def load_index(input_path):
    index, rev_index = {}, {}
    with open(input_path) as f:
        for i, line in enumerate(f.readlines()):
            v, _ = line.strip().split()
            index[v] = i
            rev_index[i] = v
    return index, rev_index


def prepare_kb_envrioment(batch_num, raw_kb_path, train_path, test_path, support_path, add_reverse_relations=True):
    """
    Process KB data which was saved as a set of triples.
        (a) Remove train and test triples from the KB envrionment.
        (b) Add reverse triples on demand.
        (c) Index unique entities and relations appeared in the KB.

    :param raw_kb_path: Path to the raw KB triples.
    :param train_path: Path to the train set KB triples.
    :param dev_path: Path to the dev set KB triples.
    :param test_path: Path to the test set KB triples.
    :param add_reverse_relations: If set, add reverse triples to the KB environment.
    """
    data_dir = os.path.dirname(raw_kb_path)

    def hist_to_vocab(_dict):
        return sorted(sorted(_dict.items(), key=lambda x: x[0]), key=lambda x: x[1], reverse=True)

    # Create entity and relation indices

    with open(train_path) as f:
        train_triples = [l.strip() for l in f.readlines()]
    test_tripless, support_tripless = [], []

    for fn in test_path:
        with open(fn) as f:
            test_tripless.append([l.strip() for l in f.readlines()])
    support_tripless.append([])
    for fn in support_path[1:]:
        with open(fn) as f:
            support_tripless.append([l.strip() for l in f.readlines()])

    entity_hist = collections.defaultdict(int)
    relation_hist = collections.defaultdict(int)
    test_triples, support_triples = [], []

    for j in range(batch_num):
        test_triples = test_triples + test_tripless[j]
        support_triples = support_triples + support_tripless[j]
    keep_triples = train_triples + support_triples
    removed_triples = test_triples

    # Index entities and relations
    for line in set(keep_triples + removed_triples):  # raw_kb_triples +
        e1, e2, r = line.strip().split()
        entity_hist[e1] += 1
        entity_hist[e2] += 1

        relation_hist[r] += 1
        if add_reverse_relations:
            inv_r = r + '_inv'
            relation_hist[inv_r] += 1
    with open(os.path.join(data_dir, 'entity2id.txt'), 'w') as o_f:
        o_f.write('{}\t{}\n'.format(DUMMY_ENTITY, DUMMY_ENTITY_ID))
        o_f.write('{}\t{}\n'.format(NO_OP_ENTITY, NO_OP_ENTITY_ID))
        for e, freq in hist_to_vocab(entity_hist):
            o_f.write('{}\t{}\n'.format(e, freq))

    with open(os.path.join(data_dir, 'relation2id.txt'), 'w') as o_f:
        o_f.write('{}\t{}\n'.format(DUMMY_RELATION, DUMMY_RELATION_ID))
        o_f.write('{}\t{}\n'.format(START_RELATION, START_RELATION_ID))
        o_f.write('{}\t{}\n'.format(NO_OP_RELATION, NO_OP_RELATION_ID))
        for r, freq in hist_to_vocab(relation_hist):
            o_f.write('{}\t{}\n'.format(r, freq))


    for i in range(-1, batch_num+1):
        if i>0:
            data_dir_ = os.path.join(data_dir, 'add_'+str(i))
        else:
            data_dir_ = data_dir
        if i != 0:
            f = open(data_dir_+'/pgrk.csv','w',encoding='utf-8',newline='')
        csv_writer = csv.writer(f)
        test_triples, support_triples = [], []
        if i > 0:
            for j in range(i):
                test_triples = test_triples + test_tripless[j]
                support_triples = support_triples + support_tripless[j]
            keep_triples = train_triples + support_triples
            removed_triples = test_triples
        else:
            keep_triples = train_triples
            removed_triples = []
        entity2id, id2entity = load_index(os.path.join(data_dir, 'entity2id.txt'))
        relation2id, id2relation = load_index(os.path.join(data_dir, 'relation2id.txt'))

        removed_triples = set(removed_triples)
        adj_list = collections.defaultdict(collections.defaultdict)
        num_facts = 0
        for line in set(keep_triples):
            e1, e2, r = line.strip().split()
            csv_writer.writerow([e1, str(1), e2, str(1)])
            csv_writer.writerow([e2, str(1), e1, str(1)])
            triple_signature = '{}\t{}\t{}'.format(e1, e2, r)
            e1_id = entity2id[e1]
            e2_id = entity2id[e2]

            if not triple_signature in removed_triples:
                r_id = relation2id[r]
                if not r_id in adj_list[e1_id]:
                    adj_list[e1_id][r_id] = set()
                if e2_id in adj_list[e1_id][r_id]:
                    print('Duplicate fact: {} ({}, {}, {})!'.format(
                        line.strip(), id2entity[e1_id], id2relation[r_id], id2entity[e2_id]))
                adj_list[e1_id][r_id].add(e2_id)
                num_facts += 1
                if add_reverse_relations:
                    inv_r = r + '_inv'
                    inv_r_id = relation2id[inv_r]
                    if not inv_r_id in adj_list[e2_id]:
                        adj_list[e2_id][inv_r_id] = set([])
                    if e1_id in adj_list[e2_id][inv_r_id]:
                        print('Duplicate fact: {} ({}, {}, {})!'.format(
                            line.strip(), id2entity[e2_id], id2relation[inv_r_id], id2entity[e1_id]))
                    adj_list[e2_id][inv_r_id].add(e1_id)
                    num_facts += 1
        print('{} facts processed'.format(num_facts))
        # Save adjacency list
        if i!=-1:
            adj_list_path = os.path.join(data_dir_, 'adj_list.pkl')
            with open(adj_list_path, 'wb') as o_f:
                pickle.dump(dict(adj_list), o_f)

