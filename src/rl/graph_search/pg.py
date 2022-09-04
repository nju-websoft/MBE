import torch
import pickle

from src.learn_framework import LFramework
import src.rl.graph_search.beam_search as search
import src.utils.ops as ops
from src.utils.ops import int_fill_var_cuda, var_cuda, zeros_var_cuda, todevice
import time
import logging


class PolicyGradient(LFramework):
    def __init__(self, args, kg, pn):
        super(PolicyGradient, self).__init__(args, kg, pn)

        # Training hyperparameters
        self.use_action_space_bucketing = args.use_action_space_bucketing
        self.num_rollouts = args.num_rollouts
        self.num_rollout_steps = args.num_rollout_steps
        self.beta = args.beta  # entropy regularization parameter
        self.gamma = args.gamma  # shrinking factor
        self.action_dropout_rate = args.action_dropout_rate
        self.action_dropout_anneal_factor = args.action_dropout_anneal_factor
        self.action_dropout_anneal_interval = args.action_dropout_anneal_interval

        # Inference hyperparameters
        self.beam_size = args.beam_size
        self.kg = kg
        self.args.attn_weight = None
        self.args.attn_reliability = None

        # Analysis
        self.path_types = dict()
        self.num_path_types = 0

        self.path_record = {i: dict() for i in range(3, kg.num_relations, 2)}
        self.path_record_attn = {i: dict() for i in range(3, kg.num_relations, 2)}
        self.path_record['candidate_path'] = {i: dict() for i in range(3, kg.num_relations, 2)}
        self.id2path, self.path2id = dict(), dict()

    def reward_fun(self, e2, pred_e2):
        return (pred_e2 == e2).float()

    def update_path_record(self, path_record, label, r):
        rule_rewards = []
        path_record_rel = torch.cat([i[0].unsqueeze(0) for i in path_record], dim=0).t().tolist()
        path_record_ent = torch.cat([i[1].unsqueeze(0) for i in path_record], dim=0).t().tolist()
        path_record_conf = torch.cat([i[2].unsqueeze(0) for i in path_record], dim=0).t().tolist()
        path_record_path = torch.cat([i[3].unsqueeze(0) for i in path_record], dim=0).t().tolist()

        def clean(task_rel, rels, ents, paths):
            res = rels[1:]  # rels[0] is the query relation
            skip = 0
            # delete loop relations (e.g., r1 -> -r1)
            for i, start in enumerate(ents):
                if skip>0:
                    skip -= 1
                    continue
                cut_l, cut_r = i, i
                for j, e in enumerate(ents[i+1:]):
                    if e == start:
                        cut_r = j+i+1
                if cut_l != cut_r:
                    skip = cut_r - cut_l
                    for x in range(cut_l, cut_r):
                        res[x] = 0
            # delete invalid relations
            path_res = []
            for r in res:
                if r>2:  # r=2 is the self-loop relation
                    path_res.append(r)
            if len(path_res) == 1 and path_res[0] == task_rel:
                for idx, r in enumerate(res):
                    if r > 2:
                        if paths[idx+1]==0:
                            break
                        return tuple(self.id2path[paths[idx+1]])
            return tuple(path_res)
        r = r.tolist()
        for rel, path_rels, path_ents, confs, paths, succ in zip(r, path_record_rel, path_record_ent, path_record_conf, path_record_path, label):
            # get path
            p = clean(rel, path_rels, path_ents, paths)
            # process path
            if succ:
                if p not in self.path_record[rel]:
                    self.path_record[rel][p] = {'pos': 0, 'neg': 0, 'conf': 0}
                    self.path_record_attn[rel][p] = {'pos':0, 'neg':0, 'conf': 0}
                self.path_record[rel][p]['pos'] += 1
                self.path_record_attn[rel][p]['pos'] += 1
            else:
                if p not in self.path_record[rel]:
                    rule_rewards.append(0)
                    continue
                self.path_record[rel][p]['neg'] += 1
                self.path_record_attn[rel][p]['neg'] += 1
            # update path info
            self.path_record[rel][p]['conf'] = self.path_record[rel][p]['pos'] / (
                    self.path_record[rel][p]['pos'] + self.path_record[rel][p]['neg'])
            self.path_record_attn[rel][p]['conf'] = self.path_record_attn[rel][p]['pos'] / (
                    self.path_record_attn[rel][p]['pos'] + self.path_record_attn[rel][p]['neg'])
            rule_rewards.append(0)
        return rule_rewards


    def get_candidate_paths(self):
        '''
        Select trustworthy paths.
        '''
        self.path_record['candidate_path'] = {i: dict() for i in range(3, self.kg.num_relations, 2)}
        self.id2path = {0: (2, 2, 2)}  # relation 2 is self-loop, (2, 2, 2) is the self-loop path
        self.path2id = {(2, 2, 2): 0}
        candidate_path_num = 0
        for rel, val in self.path_record.items():
            if rel == 'candidate_path':
                continue
            for p in val.keys():
                if val[p]['conf'] > self.args.aug_link_threshold and val[p]['pos']/float(self.train_relation_num[rel]) \
                        > self.args.aug_link_support_threshold:
                    # Note that the rollout num is 20
                    self.path_record['candidate_path'][rel][p] = self.path_record[rel][p]
                    candidate_path_num += 1
                    self.id2path[candidate_path_num] = p
                    self.path2id[p] = candidate_path_num

    def clean_path_record(self):
        for i in range(3, self.kg.num_relations, 2):
            self.path_record[i] = dict()

    def get_path_attn(self, paths):
        if not self.args.attn:
            return None
        mp_attn_value = torch.zeros(self.args.num_rel, self.args.num_rel)
        mp_attn_reliability = torch.zeros(self.kg.data_box['all']['num_rel'], self.kg.data_box['all']['num_rel'])
        original_attn_value = torch.zeros(self.args.num_rel, self.args.num_rel)
        for i, r in enumerate(range(self.kg.data_box['all']['num_rel'])):
            if r < 3 or r not in paths.keys():
                continue
            pos_total = 0
            metapaths = paths[r]
            for path in metapaths.keys():
                if paths[r][path]['pos'] < 10:
                    continue
                pos_total += paths[r][path]['pos']
                for rel in path:
                    if mp_attn_value[i, rel] < metapaths[path]['conf']:
                        mp_attn_value[i, rel] = metapaths[path]['conf']
                    if rel%2 == 1:
                        if mp_attn_value[i, rel+1] < metapaths[path]['conf']:
                            mp_attn_value[i, rel+1] = metapaths[path]['conf']
                    else:
                        if mp_attn_value[i, rel-1] < metapaths[path]['conf']:
                            mp_attn_value[i, rel-1] = metapaths[path]['conf']
            original_attn_value[i] = mp_attn_value[i]
            mp_attn_value[i] = mp_attn_value[i] * torch.tanh(torch.tensor(pos_total / 100))  # lambda=100
            mp_attn_reliability[i] = torch.tanh(torch.tensor(pos_total / 100))
        self.args.original_attn_value = original_attn_value
        self.args.attn_weight = mp_attn_value.cuda()
        self.args.attn_reliability = mp_attn_reliability.cuda()

    def save_paths(self, best=False):
        if best:
            with open(self.args.model_dir + 'path_record_best.pkl', 'wb') as f:
                pickle.dump(self.path_record, f, pickle.HIGHEST_PROTOCOL)
            with open(self.args.model_dir + 'id2path_best.pkl', 'wb') as f:
                pickle.dump(self.id2path, f, pickle.HIGHEST_PROTOCOL)
            with open(self.args.model_dir + 'path2id_best.pkl', 'wb') as f:
                pickle.dump(self.path2id, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.args.model_dir + 'path_record.pkl', 'wb') as f:
                pickle.dump(self.path_record, f, pickle.HIGHEST_PROTOCOL)
            with open(self.args.model_dir + 'id2path.pkl', 'wb') as f:
                pickle.dump(self.id2path, f, pickle.HIGHEST_PROTOCOL)
            with open(self.args.model_dir + 'path2id.pkl', 'wb') as f:
                pickle.dump(self.path2id, f, pickle.HIGHEST_PROTOCOL)

    def load_paths(self, best=False):
        if best:
            with open(self.args.model_dir + 'path_record_best.pkl', 'rb') as f:
                data = pickle.load(f)
                self.path_record = data
            with open(self.args.model_dir + 'id2path_best.pkl', 'rb') as f:
                data = self.id2path = pickle.load(f)
                self.id2path = data
            with open(self.args.model_dir + 'path2id_best.pkl', 'rb') as f:
                data = pickle.load(f)
                self.path2id = data
        else:
            with open(self.args.model_dir + 'path_record.pkl', 'rb') as f:
                self.path_record = pickle.load(f)
            with open(self.args.model_dir + 'id2path.pkl', 'rb') as f:
                self.id2path = pickle.load(f)
            with open(self.args.model_dir + 'path2id.pkl', 'rb') as f:
                self.path2id = pickle.load(f)

    def loss(self, mini_batch):
        pred_e2 = torch.LongTensor([]).cuda()
        log_action_probs = []
        action_entropy = []
        label = torch.BoolTensor([]).cuda()
        r, e1, e2 = torch.LongTensor([]).cuda(), torch.LongTensor([]).cuda(), torch.LongTensor([]).cuda()
        first_flag = True
        rules_rewards = []
        for idx, sub_mini_batch in enumerate(mini_batch):
            if len(sub_mini_batch) == 0:
                continue
            if self.args.argcn:
                self.kg.update_embedding('0')
            e1_, e2_, r_ = self.format_batch(sub_mini_batch, num_tiles=self.num_rollouts)  # sub
            output = self.rollout(e1_, r_, e2_, num_steps=self.num_rollout_steps)  # sub

            # Compute policy gradient loss
            pred_e2_ = output['pred_e2']  # sub torch.cat([], dim=0)
            pred_e2 = torch.cat([pred_e2, pred_e2_], dim=0)
            if first_flag:
                log_action_probs = output['log_action_probs']  # sub +=
                action_entropy = output['action_entropy']
            else:
                for idx2, (prob, new_prob) in enumerate(zip(log_action_probs, output['log_action_probs'])):
                    log_action_probs[idx2] = torch.cat([prob, new_prob], dim=0)
                for idx2, (entropy, new_entropy) in enumerate(zip(action_entropy, output['action_entropy'])):
                    action_entropy[idx2] = torch.cat([entropy, new_entropy], dim=0)
                # for idx2, (path_record, new_path_record) in enumerate(zip(path_record, output['path_trace'])):
                #     action_entropy[idx2] = torch.cat([path_record, new_path_record], dim=0)


            # Update path_record and confidence
              # sub +=

            label_ = (pred_e2_ == e2_)  # sub torch.cat([], dim=0)
            label = torch.cat([label, label_], dim=0)
            # self.compute_conf(path_record, r)
            r = torch.cat([r, r_], dim=0)
            e1 = torch.cat([e1, e1_], dim=0)
            e2 = torch.cat([e2, e2_], dim=0)

            rules_reward = self.update_path_record(output['path_trace'], label_, r_)
            rules_rewards += rules_reward
            first_flag = False

        # Compute discounted reward
        final_reward = self.reward_fun(e2, pred_e2) + torch.FloatTensor(rules_rewards).cuda()
        cum_discounted_rewards = [0] * self.num_rollout_steps
        cum_discounted_rewards[-1] = final_reward
        R = 0
        for i in range(self.num_rollout_steps - 1, -1, -1):
            R = self.gamma * R + cum_discounted_rewards[i]
            cum_discounted_rewards[i] = R

        # Compute policy gradient
        pg_loss, pt_loss = 0, 0
        for i in range(self.num_rollout_steps):
            log_action_prob = log_action_probs[i]
            pg_loss += -cum_discounted_rewards[i] * log_action_prob
            pt_loss += -cum_discounted_rewards[i] * torch.exp(log_action_prob)

        # Entropy regularization
        entropy = torch.cat([x.unsqueeze(1) for x in action_entropy], dim=1).mean(dim=1)
        pg_loss = (pg_loss - entropy * self.beta).mean()
        pt_loss = (pt_loss - entropy * self.beta).mean()

        loss_dict = {}
        loss_dict['model_loss'] = pg_loss
        loss_dict['print_loss'] = float(pt_loss)
        loss_dict['reward'] = final_reward
        loss_dict['entropy'] = float(entropy.mean())

        return loss_dict

    def rollout(self, e_s, q, e_t, num_steps, visualize_action_probs=False):
        """
        Perform multi-step rollout from the source entity conditioned on the query relation.
        :param pn: Policy network.
        :param e_s: (Variable:batch) source entity indices.
        :param q: (Variable:batch) query relation indices.
        :param e_t: (Variable:batch) target entity indices.
        :param kg: Knowledge graph environment.
        :param num_steps: Number of rollout steps.
        :param visualize_action_probs: If set, save action probabilities for visualization.
        :return pred_e2: Target entities reached at the end of rollout.
        :return log_path_prob: Log probability of the sampled path.
        :return action_entropy: Entropy regularization term.
        """
        assert (num_steps > 0)
        kg, pn = self.kg, self.mdl

        # Initialization
        log_action_probs = []
        action_entropy = []
        r_s = int_fill_var_cuda(e_s.size(), kg.dummy_start_r)
        seen_nodes = int_fill_var_cuda(e_s.size(), kg.dummy_e).unsqueeze(1)
        path_components = []

        path_trace = [(r_s, e_s, todevice([1.0 for i in range(len(r_s))]), todevice([1 for i in range(len(r_s))]).long())]
        pn.initialize_path((r_s, e_s, todevice([1.0 for i in range(len(r_s))]), todevice([1 for i in range(len(r_s))]).long()), kg)

        for t in range(num_steps):
            last_r, e, c, p = path_trace[-1]
            obs = [e_s, q, e_t, t==(num_steps-1), last_r, seen_nodes]
            db_outcomes, inv_offset, policy_entropy = pn.transit(
                e, obs, kg, use_action_space_bucketing=self.use_action_space_bucketing)
            sample_outcome = self.sample_action(db_outcomes, inv_offset)
            action = sample_outcome['action_sample']
            pn.update_path(action, kg)
            action_prob = sample_outcome['action_prob']
            log_action_probs.append(ops.safe_log(action_prob))
            action_entropy.append(policy_entropy)
            seen_nodes = torch.cat([seen_nodes, e.unsqueeze(1)], dim=1)
            path_trace.append(action)

            if visualize_action_probs:
                top_k_action = sample_outcome['top_actions']
                top_k_action_prob = sample_outcome['top_action_probs']
                path_components.append((e, top_k_action, top_k_action_prob))

        pred_e2 = path_trace[-1][1]

        return {
            'pred_e2': pred_e2,
            'log_action_probs': log_action_probs,
            'action_entropy': action_entropy,
            'path_trace': path_trace,
            'path_components': path_components
        }

    def sample_action(self, db_outcomes, inv_offset=None):
        """
        Sample an action based on current policy.
        :param db_outcomes (((r_space, e_space), action_mask), action_dist):
                r_space: (Variable:batch) relation space
                e_space: (Variable:batch) target entity space
                action_mask: (Variable:batch) binary mask indicating padding actions.
                action_dist: (Variable:batch) action distribution of the current step based on set_policy
                    network parameters
        :param inv_offset: Indexes for restoring original order in a batch.
        :return next_action (next_r, next_e): Sampled next action.
        :return action_prob: Probability of the sampled action.
        """

        def apply_action_dropout_mask(action_dist, action_mask):
            if self.action_dropout_rate > 0:
                rand = torch.rand(action_dist.size())
                # print('rand',rand, '>', rand > self.action_dropout_rate)
                action_keep_mask = var_cuda(rand > self.action_dropout_rate).float()
                # There is a small chance that that action_keep_mask is accidentally set to zero.
                # When this happen, we take a random sample from the available actions.
                # sample_action_dist = action_dist * (action_keep_mask + ops.EPSILON)
                sample_action_dist = \
                    action_dist * action_keep_mask + ops.EPSILON * (1 - action_keep_mask) * action_mask
                return sample_action_dist
            else:
                return action_dist

        def sample(action_space, action_dist):
            sample_outcome = {}
            ((r_space, e_space, c_space, p_space), action_mask) = action_space
            sample_action_dist = apply_action_dropout_mask(action_dist, action_mask)
            try:
                idx = torch.multinomial(sample_action_dist, 1, replacement=True)
            except:
                print(sample_action_dist)
            next_r = ops.batch_lookup(r_space, idx)
            next_e = ops.batch_lookup(e_space, idx)
            next_c = ops.batch_lookup(c_space, idx)
            next_p = ops.batch_lookup(p_space, idx)
            action_prob = ops.batch_lookup(action_dist, idx)
            sample_outcome['action_sample'] = (next_r, next_e, next_c, next_p)
            sample_outcome['action_prob'] = action_prob
            return sample_outcome

        if inv_offset is not None:
            next_r_list = []
            next_e_list = []
            next_c_list = []
            next_p_list = []
            action_dist_list = []
            action_prob_list = []
            for action_space, action_dist in db_outcomes:
                sample_outcome = sample(action_space, action_dist)
                next_r_list.append(sample_outcome['action_sample'][0])
                next_e_list.append(sample_outcome['action_sample'][1])
                next_c_list.append(sample_outcome['action_sample'][2])
                next_p_list.append(sample_outcome['action_sample'][3])
                action_prob_list.append(sample_outcome['action_prob'])
                action_dist_list.append(action_dist)
            next_r = torch.cat(next_r_list, dim=0)[inv_offset]
            next_e = torch.cat(next_e_list, dim=0)[inv_offset]
            next_c = torch.cat(next_c_list, dim=0)[inv_offset]
            next_p = torch.cat(next_p_list, dim=0)[inv_offset]
            action_sample = (next_r, next_e, next_c, next_p)
            action_prob = torch.cat(action_prob_list, dim=0)[inv_offset]
            sample_outcome = {}
            sample_outcome['action_sample'] = action_sample
            sample_outcome['action_prob'] = action_prob
        else:
            sample_outcome = sample(db_outcomes[0][0], db_outcomes[0][1])

        return sample_outcome

    def predict(self, mini_batch, verbose=False):
        kg, pn = self.kg, self.mdl
        e1, e2, r = self.format_batch(mini_batch)
        beam_search_output = search.beam_search(
            pn, e1, r, e2, kg, self.num_rollout_steps, self.beam_size)
        pred_e2s = beam_search_output['pred_e2s']
        pred_e2_scores = beam_search_output['pred_e2_scores']
        if verbose:
            # print inference paths
            search_traces = beam_search_output['search_traces']
            output_beam_size = min(self.beam_size, pred_e2_scores.shape[1])
            for i in range(len(e1)):
                for j in range(output_beam_size):
                    ind = i * output_beam_size + j
                    if pred_e2s[i][j] == kg.dummy_e:
                        break
                    search_trace = []
                    for k in range(len(search_traces)):
                        search_trace.append((int(search_traces[k][0][ind]), int(search_traces[k][1][ind])))
                    logging.info('beam {}: score = {} \n<PATH> {}'.format(
                        j, float(pred_e2_scores[i][j]), ops.format_path(search_trace, kg)))
        with torch.no_grad():
            pred_scores = zeros_var_cuda([len(e1), kg.num_entities])
            for i in range(len(e1)):
                pred_scores[i][pred_e2s[i]] = torch.exp(pred_e2_scores[i])
        return pred_scores
