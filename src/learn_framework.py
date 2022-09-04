import os
import random
import shutil
from tqdm import tqdm
from data_utils import load_triples
from prettytable import PrettyTable
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import src.eval
from src.utils.ops import var_cuda, zeros_var_cuda
import src.utils.ops as ops
import logging


class LFramework(nn.Module):
    def __init__(self, args, kg, mdl):
        super(LFramework, self).__init__()
        self.args = args
        self.data_dir = args.data_dir
        self.model_dir = args.model_dir
        self.model = args.model

        # Training hyperparameters
        self.batch_size = args.batch_size
        self.train_batch_size = args.train_batch_size
        self.dev_batch_size = args.dev_batch_size
        self.start_epoch = args.start_epoch
        self.num_epochs = args.num_epochs
        self.num_wait_epochs = args.num_wait_epochs
        self.num_peek_epochs = args.num_peek_epochs
        self.learning_rate = args.learning_rate
        self.grad_norm = args.grad_norm
        self.adam_beta1 = args.adam_beta1
        self.adam_beta2 = args.adam_beta2
        self.optim = None

        self.inference = not args.train
        self.args.attn_weight = None
        self.args.attn_reliability = None

        self.kg = kg
        self.mdl = mdl
        logging.info('{} module created'.format(self.model))

    def print_all_model_parameters(self):
        logging.info('\nModel Parameters')
        logging.info('--------------------------')
        for name, param in self.named_parameters():
            logging.info((name, param.numel(), 'requires_grad={}'.format(param.requires_grad)))
        param_sizes = [param.numel() for param in self.parameters()]
        logging.info('Total # parameters = {}'.format(sum(param_sizes)))
        logging.info('--------------------------')

    def run_train(self, train_data):
        self.print_all_model_parameters()

        if self.optim is None:
            self.optim = optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)

        # Track dev metrics changes
        best_dev_metrics = 0
        dev_metrics_history = []

        self.train_relation_num = {i:0 for i in range(self.args.num_rel)}
        for triple in train_data:
            h, t, r = triple
            self.train_relation_num[r] += 1

        # Start training
        for epoch_id in range(self.start_epoch, self.num_epochs):
            train_relation = None
            # If use relation-aware feedback attention, preprocess for train data
            if self.args.attn:
                # To save memory, group examples by query relation
                relations = [i for i in range(self.args.num_rel)]
                train_data__ = train_data
                random.shuffle(train_data__)
                train_relation = [[] for i in range(self.args.num_rel)]
                for triple in train_data__:
                    train_relation[triple[2]].append(triple)

                train_group_data = []
                mix_data = []
                random.shuffle(relations)

                for rel in relations:
                    if train_relation[rel] == []:
                        continue
                    for example_id in range(0, len(train_relation[rel]), self.batch_size):
                        if example_id + self.batch_size <= len(train_relation[rel]):
                            train_group_data.append([train_relation[rel][example_id:example_id + self.batch_size]])
                        else:
                            mix_data.append(train_relation[rel][example_id:example_id + self.batch_size])
                length = self.batch_size
                now_batch = []
                while len(mix_data) > 0:
                    small_group = mix_data.pop()
                    if len(small_group) < length:
                        now_batch.append(small_group)
                        length -= len(small_group)
                    else:
                        now_batch.append(small_group[:length])
                        train_group_data.append(now_batch)
                        now_batch = [small_group[length:]]
                        length = self.batch_size - len(small_group[length:])
                if len(now_batch) > 0:
                    train_group_data.append(now_batch)
            else:
                train_group_data = []
                for example_id in range(0, len(train_data), self.batch_size):
                    train_group_data.append([train_data[example_id:example_id + self.batch_size]])

            self.kg.prepare_new_batch('0')
            # group training samples for saving memory
            train_data_ = []
            idx = [i for i in range(len(train_group_data))]
            random.shuffle(idx)
            for i in idx:
                train_data_.append(train_group_data[i])
            logging.info('Epoch {}'.format(epoch_id))

            # Update model parameters
            self.train()

            random.shuffle(train_data_)
            batch_losses = []
            entropies = []

            for mini_batch in tqdm(train_data_):
                self.optim.zero_grad()
                if self.args.attn:
                    self.get_path_attn(self.path_record)
                loss = self.loss(mini_batch)
                loss['model_loss'].backward()
                if self.grad_norm > 0:
                    clip_grad_norm_(self.parameters(), self.grad_norm)

                self.optim.step()

                batch_losses.append(loss['print_loss'])

                entropies.append(loss['entropy'])
            # Check training statistics
            stdout_msg = 'Epoch {}: average training loss = {}'.format(epoch_id, np.mean(batch_losses))
            if entropies:
                stdout_msg += ' entropy = {}'.format(np.mean(entropies))
            logging.info(stdout_msg)
            self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id)
            self.save_paths()

            torch.cuda.empty_cache()

            # Check valid set performance
            if epoch_id > 0 and epoch_id % self.num_peek_epochs == 0:
                pt = PrettyTable()
                pt.field_names = ['Batch', 'Hits@1', 'Hits@3', 'Hits@5', 'Hits@10', 'MRR']
                torch.cuda.empty_cache()
                batch_id = str(1)
                self.args.now_batch = batch_id
                self.kg.prepare_new_batch(batch_id)
                if self.args.aug_link:
                    self.kg.aug_link_info = self.kg.add_aug_links(self.path_record['candidate_path'],
                                                                  self.id2path, self.path2id,
                                                                  mode='test')
                    if self.kg.aug_link_info is not None:
                        # add augmentation links to action space
                        self.kg.vectorize_action_space_aug_link(self.kg.aug_link_info)
                self.eval()

                entity_index_path = os.path.join(self.args.data_dir, 'entity2id.txt')
                relation_index_path = os.path.join(self.args.data_dir, 'relation2id.txt')
                test_path = os.path.join(self.args.data_dir, 'add_' + batch_id + '/' + 'valid.triples')
                test_data_ = load_triples(test_path, entity_index_path, relation_index_path,
                                          seen_entities=set())
                dev_score, using_examples = self.forward(test_data_)
                h_1, h_3, h_5, h_10, mrr = src.eval.hits_and_ranks(using_examples, dev_score, self.kg.all_objects,
                                                                   verbose=False)
                metrics = mrr
                pt.add_row([str(int(batch_id)-1), round(h_1, 3), round(h_3, 3), round(h_5, 3), round(h_10, 3), round(mrr, 3)])
                torch.cuda.empty_cache()

                # if run analysis, evaluate both validation and test data
                if self.args.run_analysis:
                    for i in range(1, self.args.batch_num):
                        torch.cuda.empty_cache()
                        batch_id = str(i+1)
                        self.args.now_batch = batch_id
                        self.kg.prepare_new_batch(batch_id)
                        if self.args.aug_link:
                            self.kg.aug_link_info = self.kg.add_aug_links(self.path_record['candidate_path'],
                                                                              self.id2path, self.path2id,
                                                                              mode='test')
                            if self.kg.aug_link_info is not None:
                                self.kg.vectorize_action_space_aug_link(self.kg.aug_link_info)
                        self.eval()

                        entity_index_path = os.path.join(self.args.data_dir, 'entity2id.txt')
                        relation_index_path = os.path.join(self.args.data_dir, 'relation2id.txt')
                        test_path = os.path.join(self.args.data_dir, 'add_'+batch_id+'/'+'test.triples')
                        test_data_ = load_triples(test_path, entity_index_path, relation_index_path,
                                                                 seen_entities=set())
                        dev_score, using_examples = self.forward(test_data_)
                        h_1, h_3, h_5, h_10, mrr = src.eval.hits_and_ranks(using_examples, dev_score, self.kg.all_objects,
                                                                  verbose=False)
                        pt.add_row([str(int(batch_id)-1), round(h_1,3), round(h_3,3), round(h_5,3), round(h_10,3), round(mrr,3)])
                        torch.cuda.empty_cache()
                logging.info(pt)
                # Action dropout anneaking
                if self.model.startswith('point'):
                    eta = self.action_dropout_anneal_interval
                    if len(dev_metrics_history) > eta and metrics < min(dev_metrics_history[-eta:]):
                        old_action_dropout_rate = self.action_dropout_rate
                        self.action_dropout_rate *= self.action_dropout_anneal_factor 
                        logging.info('Decreasing action dropout rate: {} -> {}'.format(
                            old_action_dropout_rate, self.action_dropout_rate))

                # Save checkpoint
                if metrics > best_dev_metrics:
                    self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id, is_best=True)
                    self.save_paths(best=True)
                    best_dev_metrics = metrics
                    with open(os.path.join(self.model_dir, 'best_dev_iteration.dat'), 'w') as o_f:
                        o_f.write('{}'.format(epoch_id))
                else:
                    # Early stopping
                    if epoch_id >= self.num_wait_epochs and metrics < np.mean(dev_metrics_history[-self.num_wait_epochs:]):
                        break
                dev_metrics_history.append(metrics)

            if self.args.aug_link:
                self.args.now_batch = '0'
                self.kg.prepare_new_batch('0')
                self.get_candidate_paths()
                self.kg.aug_link_info = self.kg.add_aug_links(self.path_record['candidate_path'], self.id2path, self.path2id)
                if self.kg.aug_link_info is not None:
                    self.kg.vectorize_action_space_aug_link(self.kg.aug_link_info)
                    self.clean_path_record()
            train_relation.clear()
            train_data_.clear()

    def forward(self, examples, verbose=False):
        if self.args.attn:
            self.get_path_attn(self.path_record)
        pred_scores = []
        using_examples = []
        # to save memory, group test set according to query relation
        if self.args.attn:
            test_relation = [[] for i in range(self.args.num_rel)]
            for triple in examples:
                test_relation[triple[2]].append(triple)
            test_group_data = []
            for rel in range(self.args.num_rel):
                if test_relation[rel] == []:
                    continue
                for example_id in range(0, len(test_relation[rel]), self.dev_batch_size):
                    test_group_data.append(test_relation[rel][example_id:example_id + self.dev_batch_size])
        else:
            test_group_data = []
            for example_id in range(0, len(examples), self.dev_batch_size):
                test_group_data.append(examples[example_id:example_id + self.dev_batch_size])

        for mini_batch in tqdm(test_group_data):
            mini_batch_size = len(mini_batch)
            if len(mini_batch) < self.dev_batch_size:
                self.make_full_batch(mini_batch, self.dev_batch_size)
            if self.args.argcn and self.args.attn:
                self.get_path_attn(self.path_record)
                self.kg.update_embedding(self.args.now_batch, query=mini_batch[0][2])
            pred_score = self.predict(mini_batch, verbose=verbose)
            pred_scores.append(pred_score[:mini_batch_size])
            using_examples += mini_batch[:mini_batch_size]
        scores = torch.cat(pred_scores)

        return scores, using_examples

    def format_batch(self, batch_data, num_labels=-1, num_tiles=1):
        """
        Convert batched tuples to the tensors accepted by the NN.
        """
        def convert_to_binary_multi_subject(e1):
            e1_label = zeros_var_cuda([len(e1), num_labels])
            for i in range(len(e1)):
                e1_label[i][e1[i]] = 1
            return e1_label

        def convert_to_binary_multi_object(e2):
            e2_label = zeros_var_cuda([len(e2), num_labels])
            for i in range(len(e2)):
                e2_label[i][e2[i]] = 1
            return e2_label

        batch_e1, batch_e2, batch_r = [], [], []
        for i in range(len(batch_data)):
            e1, e2, r = batch_data[i]
            batch_e1.append(e1)
            batch_e2.append(e2)
            batch_r.append(r)
        batch_e1 = var_cuda(torch.LongTensor(batch_e1), requires_grad=False)
        batch_r = var_cuda(torch.LongTensor(batch_r), requires_grad=False)
        if type(batch_e2[0]) is list:
            batch_e2 = convert_to_binary_multi_object(batch_e2)
        elif type(batch_e1[0]) is list:
            batch_e1 = convert_to_binary_multi_subject(batch_e1)
        else:
            batch_e2 = var_cuda(torch.LongTensor(batch_e2), requires_grad=False)
        # Rollout multiple times for each example
        if num_tiles > 1:
            batch_e1 = ops.tile_along_beam(batch_e1, num_tiles)
            batch_r = ops.tile_along_beam(batch_r, num_tiles)
            batch_e2 = ops.tile_along_beam(batch_e2, num_tiles)
        return batch_e1, batch_e2, batch_r

    def make_full_batch(self, mini_batch, batch_size, multi_answers=False):
        dummy_e = self.kg.dummy_e
        dummy_r = self.kg.dummy_r
        if multi_answers:
            dummy_example = (dummy_e, [dummy_e], dummy_r)
        else:
            dummy_example = (dummy_e, dummy_e, dummy_r)
        for _ in range(batch_size - len(mini_batch)):
            mini_batch.append(dummy_example)

    def save_checkpoint(self, checkpoint_id, epoch_id=None, is_best=False):
        """
        Save model checkpoint.
        :param checkpoint_id: Model checkpoint index assigned by training loop.
        :param epoch_id: Model epoch index assigned by training loop.
        :param is_best: if set, the model being saved is the best model on dev set.
        """
        checkpoint_dict = dict()
        checkpoint_dict['state_dict'] = self.state_dict()
        checkpoint_dict['epoch_id'] = epoch_id

        out_tar = os.path.join(self.model_dir, 'checkpoint-{}.tar'.format(checkpoint_id))
        if is_best:
            best_path = os.path.join(self.model_dir, 'model_best.tar')
            shutil.copyfile(out_tar, best_path)
            logging.info('=> best model updated \'{}\''.format(best_path))
        else:
            torch.save(checkpoint_dict, out_tar)
            logging.info('=> saving checkpoint to \'{}\''.format(out_tar))

    def load_checkpoint(self, input_file):
        """
        Load model checkpoint.
        :param input_file: Checkpoint file path.
        """
        if os.path.isfile(input_file):
            logging.info('=> loading checkpoint \'{}\''.format(input_file))
            checkpoint = torch.load(input_file, map_location="cuda:{}".format(self.args.gpu))
            # print(checkpoint['state_dict'])
            checkpoint['state_dict'].pop('kg.relation_embeddings')
            self.load_state_dict(checkpoint['state_dict'])
            if not self.inference:
                self.start_epoch = checkpoint['epoch_id'] + 1
                assert (self.start_epoch <= self.num_epochs)
        else:
            logging.info('=> no checkpoint found at \'{}\''.format(input_file))


    @property
    def rl_variation_tag(self):
        parts = self.model.split('.')
        if len(parts) > 1:
            return parts[1]
        else:
            return ''
