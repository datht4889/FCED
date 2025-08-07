import pickle
import os 
import random
import numpy as np
from transformers import BertTokenizer, RobertaTokenizer, enable_full_determinism

class data_sampler_CFRL(object):
    def __init__(self, config=None, seed=None):
        self.config = config
        self.max_length = self.config.max_length
        self.task_length = self.config.task_length
        self.unused_tokens = ['[unused0]', '[unused1]', '[unused2]', '[unused3]']
        self.unused_token = '[unused0]'
        if self.config.model == 'bert':
            self.mask_token = '[MASK]' 
            model_path = self.config.bert_path
            tokenizer_from_pretrained = BertTokenizer.from_pretrained
        elif self.config.model == 'roberta':
            self.mask_token = '<mask>' 
            model_path = self.config.roberta_path
            tokenizer_from_pretrained = RobertaTokenizer.from_pretrained

        # tokenizer
        if config.pattern == 'marker':
            self.tokenizer = tokenizer_from_pretrained(model_path, \
            additional_special_tokens=self.unused_tokens)
            self.config.h_ids = self.tokenizer.get_vocab()[self.unused_tokens[0]]
            self.config.t_ids = self.tokenizer.get_vocab()[self.unused_tokens[2]]
        elif config.pattern == 'hardprompt' or config.pattern == 'cls':
            self.tokenizer = tokenizer_from_pretrained(model_path)
        elif config.pattern == 'softprompt' or config.pattern == 'hybridprompt':
            self.tokenizer =tokenizer_from_pretrained(model_path, \
            additional_special_tokens=[self.unused_token])
            self.config.prompt_token_ids = self.tokenizer.get_vocab()[self.unused_token]

        self.config.vocab_size = len(self.tokenizer)
        self.config.sep_token_ids = self.tokenizer.get_vocab()[self.tokenizer.sep_token]
        self.config.mask_token_ids = self.tokenizer.get_vocab()[self.tokenizer.mask_token]
        self.sep_token_ids, self.mask_token_ids =  self.config.sep_token_ids, self.config.mask_token_ids

        # read relations
        self.id2rel, self.rel2id = self._read_relations(self.config.relation_name)
        self.rel2des, self.id2des = self._read_descriptions(self.config.relation_description)

        self.config.num_of_relation = len(self.id2rel)

        # read data
        self.training_data = self._read_data(self.config.training_data, self._temp_datapath('train'))
        self.valid_data = self._read_data(self.config.valid_data, self._temp_datapath('valid'))
        self.test_data = self._read_data(self.config.test_data, self._temp_datapath('test'))

        # read relation order
        rel_index = np.load(self.config.rel_index)
        rel_cluster_label = np.load(self.config.rel_cluster_label)
        self.cluster_to_labels = {}
        for index, i in enumerate(rel_index):
            if rel_cluster_label[index] in self.cluster_to_labels.keys():
                self.cluster_to_labels[rel_cluster_label[index]].append(i-1)
            else:
                self.cluster_to_labels[rel_cluster_label[index]] = [i-1]

        # shuffle task order
        self.seed = seed
        if self.seed != None:
            self.set_seed(self.seed)

        self.shuffle_index_old = list(range(self.task_length - 1))
        random.shuffle(self.shuffle_index_old)
        self.shuffle_index_old = np.argsort(self.shuffle_index_old)
        self.shuffle_index = np.insert(self.shuffle_index_old, 0, self.task_length - 1)        
        print(f'Task_order: {self.shuffle_index}')
        self.batch = 0

        # record relations
        self.seen_relations = []
        self.history_test_data = {}
        self.seen_descriptions = {}


    def set_seed(self, seed):
        self.seed = seed
        if self.seed != None:
            # random.seed(self.seed)
            enable_full_determinism(self.seed)
        self.shuffle_index_old = list(range(self.task_length - 1))
        random.shuffle(self.shuffle_index_old)
        self.shuffle_index_old = np.argsort(self.shuffle_index_old)
        self.shuffle_index = np.insert(self.shuffle_index_old, 0, self.task_length - 1)


    def __iter__(self):
        return self

    def __next__(self):
        if self.batch == self.task_length:
            raise StopIteration()
        
        indexs = self.cluster_to_labels[self.shuffle_index[self.batch]]
        self.batch += 1

        current_relations = []
        cur_training_data = {}
        cur_valid_data = {}
        cur_test_data = {}

        for index in indexs:
            current_relations.append(self.id2rel[index])
            self.seen_relations.append(self.id2rel[index])

            cur_training_data[self.id2rel[index]] = self.training_data[index]
            cur_valid_data[self.id2rel[index]] = self.valid_data[index]
            cur_test_data[self.id2rel[index]] = self.test_data[index]
            self.history_test_data[self.id2rel[index]] = self.test_data[index]
            # fix_here 
            self.seen_descriptions[self.id2rel[index]] = self.id2des[index]
        return cur_training_data, cur_valid_data, cur_test_data, current_relations,\
            self.history_test_data, self.seen_relations, self.seen_descriptions

    def _temp_datapath(self, data_type):
        '''
            data_type = 'train'/'valid'/'test'
        '''
        temp_name = [data_type]
        file_name = '{}.pkl'.format('-'.join([str(x) for x in temp_name]))
        prompt_len = self.config.prompt_len * self.config.prompt_num
        if self.config.model == 'bert':
            tp1 = '_process_BERT_'
        elif self.config.model == 'roberta':
            tp1 = '_process_Roberta_'
        if self.config.task_name == 'FewRel':
            tp2 = 'CFRLFewRel/CFRLdata_10_100_10_'
        else:
            tp2 = 'CFRLTacred/CFRLdata_6_100_5_'
        if self.config.pattern == 'hardprompt':
            mid_dir = os.path.join('data', tp2 + str(self.config.num_k), \
            tp1  + self.config.pattern)
        elif self.config.pattern == 'softprompt' or self.config.pattern == 'hybridprompt':                
            mid_dir = os.path.join('data', tp2 + str(self.config.num_k), \
            tp1 + self.config.pattern + '_' + str(prompt_len) + 'token')
        elif self.config.pattern == 'cls':
            mid_dir = os.path.join('data', tp2 + str(self.config.num_k), \
            tp1 + self.config.pattern)            
        elif self.config.pattern == 'marker':
            mid_dir = os.path.join('data', tp2 + str(self.config.num_k),  \
            tp1 + self.config.pattern)      
        if not os.path.exists(mid_dir):
            os.mkdir(mid_dir)
        save_data_path = os.path.join(mid_dir, file_name)   
        return save_data_path     

    def _read_data(self, file, save_data_path):
        if os.path.isfile(save_data_path):
            with open(save_data_path, 'rb') as f:
                datas = pickle.load(f)
                print(save_data_path)
            return datas
        else:
            samples = []
            with open(file) as f:
                for i, line in enumerate(f):
                    sample = {}
                    items = line.strip().split('\t')
                    if (len(items[0]) > 0):
                        sample['relation'] = int(items[0]) - 1
                        sample['index'] = i
                        if items[1] != 'noNegativeAnswer':
                            candidate_ixs = [int(ix) for ix in items[1].split()]
                            sample['tokens'] = items[2].split()
                            sample['description'] = self.id2des[sample['relation']]
                            headent = items[3]
                            headidx = [[int(ix) for ix in items[4].split()]]
                            tailent = items[5]
                            tailidx = [[int(ix) for ix in items[6].split()]]
                            headid = items[7]
                            tailid = items[8]
                            sample['h'] = [headent, headid, headidx]
                            sample['t'] = [tailent, tailid, tailidx]
                            samples.append(sample)

            read_data = [[] for i in range(self.config.num_of_relation)]
            for sample in samples:
                tokenized_sample = self.tokenize(sample)
                read_data[tokenized_sample['relation']].append(tokenized_sample)
            with open(save_data_path, 'wb') as f:
                pickle.dump(read_data, f)
                print(save_data_path)
            return read_data

    def tokenize(self, sample):
        tokenized_sample = {}
        tokenized_sample['relation'] = sample['relation']
        tokenized_sample['index'] = sample['index']
        if self.config.pattern == 'hardprompt':
            ids, mask = self._tokenize_hardprompt(sample)
        elif self.config.pattern == 'softprompt':
            ids, mask = self._tokenize_softprompt(sample)   
        elif self.config.pattern == 'hybridprompt':
            ids, mask = self._tokenize_hybridprompt(sample)                     
        elif self.config.pattern == 'marker':
            ids, mask = self._tokenize_marker(sample)
        elif self.config.pattern == 'cls':
            ids, mask = self._tokenize_cls(sample)            
        tokenized_sample['ids'] = ids
        tokenized_sample['mask'] = mask    
        return tokenized_sample    


    def _read_relations(self, file):
        id2rel, rel2id = {}, {}
        with open(file) as f:
            for index, line in enumerate(f):
                rel = line.strip()
                id2rel[index] = rel
                rel2id[rel] = index
        return id2rel, rel2id
    
    def _read_descriptions(self, file):
            # id2rel, rel2id = {}, {}
            rel2des = {}
            id2des = {}
            with open(file) as f:
                for index, line in enumerate(f):
                    rel = line.strip()
                    x = rel.split('\t')
                    rel2des[x[1]] = x[2]
                    id2des[int(x[0])] = x[2]
            return rel2des, id2des 
    
    def _tokenize_softprompt(self, sample):
        '''
        X [v] [v] [v] [v]
        [v] = [unused0] * prompt_len
        '''
        prompt_len = self.config.prompt_len
        raw_tokens = sample['tokens']
        prompt = raw_tokens + [self.unused_token] * prompt_len + [self.unused_token] * prompt_len \
                             + [self.unused_token] * prompt_len + [self.unused_token] * prompt_len  
        ids = self.tokenizer.encode(' '.join(prompt),
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.max_length)        
        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
        mask[:end_index + 1] = 1 

        return ids, mask           

    def _tokenize_hybridprompt(self, sample):
        '''
        X [v] e1 [v] [MASK] [v] e2 [v] 
        [v] = [unused0] * prompt_len
        '''
        prompt_len = self.config.prompt_len
        raw_tokens = sample['tokens']
        h, t = sample['h'][0].split(' '),  sample['t'][0].split(' ')
        prompt = raw_tokens + [self.unused_token] * prompt_len + h + [self.unused_token] * prompt_len \
               + [self.mask_token] + [self.unused_token] * prompt_len + t + [self.unused_token] * prompt_len  
        ids = self.tokenizer.encode(' '.join(prompt),
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.max_length)        
        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
        mask[:end_index + 1] = 1 

        return ids, mask        

    def _tokenize_hardprompt(self, sample):
        '''
        X e1 [MASK] e2 
        '''
        raw_tokens = sample['tokens']
        h, t = sample['h'][0].split(' '),  sample['t'][0].split(' ')
        prompt = raw_tokens +  h + [self.mask_token] + t
        ids = self.tokenizer.encode(' '.join(prompt),
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.max_length)
        
        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
        mask[:end_index + 1] = 1 

        return ids, mask

    def _tokenize_marker(self, sample):
        '''
        [unused]e[unused]
        '''
        raw_tokens = sample['tokens']
        h1, h2, t1, t2 =  sample['h'][2][0][0], sample['h'][2][0][-1], sample['t'][2][0][0], sample['t'][2][0][-1]
        new_tokens = []

        # add entities marker        
        for index, token in enumerate(raw_tokens):
            if index == h1:
                new_tokens.append(self.unused_tokens[0])
                new_tokens.append(token)
                if index == h2:
                    new_tokens.append(self.unused_tokens[1])
            elif index == h2:
                new_tokens.append(token)
                new_tokens.append(self.unused_tokens[1])
            elif index == t1:
                new_tokens.append(self.unused_tokens[2])
                new_tokens.append(token)
                if index == t2:
                    new_tokens.append(self.unused_tokens[3])
            elif index == t2:
                new_tokens.append(token)
                new_tokens.append(self.unused_tokens[3])
            else:
                new_tokens.append(token)
            
            ids = self.tokenizer.encode(' '.join(new_tokens),
                                        padding='max_length',
                                        truncation=True,
                                        max_length=self.max_length)
            
            # mask
            mask = np.zeros(self.max_length, dtype=np.int32)
            end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
            mask[:end_index + 1] = 1

        return ids, mask

    def _tokenize_cls(self, sample):
        '''
        [CLS] X
        '''
        raw_tokens = sample['tokens']
        ids = self.tokenizer.encode(' '.join(raw_tokens),
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.max_length)
        
        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
        mask[:end_index + 1] = 1

        return ids, mask
    

if __name__ == "__main__":
    import argparse
    from config import Config
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="Tacred", type=str)
    parser.add_argument("--use_llm", action = 'store_true', default=False)
    parser.add_argument("--num_k", default=5, type=int) # 5
    parser.add_argument("--num_gen", default=5, type=int) # 5 
    parser.add_argument("--mixup", action = 'store_true')
    parser.add_argument("--epoch", default=8, type=int) # 8, 10
    parser.add_argument("--epoch_mem", default=6, type=int) # 6, 10
    parser.add_argument("--mixup_loss_1", default=0.25, type=float) # 0.25, 0.5
    parser.add_argument("--mixup_loss_2", default=0.25, type=float) # 0.25, 0.5
    parser.add_argument("--base_optimizer", default="AdamW", type=str)
    parser.add_argument("--SAM", action = 'store_true', default=False)
    parser.add_argument("--sam_optimizer", default="SAM", type=str)
    parser.add_argument("--SAM_type", default="current", type=str)
    parser.add_argument("--rho", default=0.05, type=float)
    parser.add_argument("--decay", default=0, type=float)

    args = parser.parse_args()
    if args.use_llm:
        config = Config('config_llm.ini')
    else:
        config = Config('config.ini')
    config.task_name = args.task_name
    config.use_llm = args.use_llm
    config.num_k = args.num_k
    config.num_gen = args.num_gen
    config.mixup = args.mixup
    config.epoch = args.epoch
    config.epoch_mem = args.epoch_mem
    config.mixup_loss_1 = args.mixup_loss_1
    config.mixup_loss_2 = args.mixup_loss_2

    config.base_optimizer = args.base_optimizer
    config.SAM = args.SAM
    config.SAM_type = args.SAM_type
    config.rho = args.rho
    config.sam_optimizer = args.sam_optimizer
    config.decay = args.decay

    print("CPL Start")
    print(f'task_name: {config.task_name}')
    print(f'mixup: {config.mixup}')
    print(f'base_optimizer: {config.base_optimizer}')
    print(f'SAM: {config.SAM}')
    print(f'SAM_type: {config.SAM_type}')
    print(f'SAM Optimizer: {config.sam_optimizer}')
    print(f'decay: {config.decay}')

    if config.task_name == 'FewRel':
        config.rel_index = './data/CFRLFewRel/rel_index.npy'
        config.relation_name = './data/CFRLFewRel/relation_name.txt'
        config.relation_description = './data/CFRLFewRel/relation_description.txt'
        if config.num_k == 5:
            config.rel_cluster_label = './data/CFRLFewRel/CFRLdata_10_100_10_5/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/train_0.txt'
            config.valid_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/valid_0.txt'
            config.test_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/test_0.txt'
        elif config.num_k == 10:
            config.rel_cluster_label = './data/CFRLFewRel/CFRLdata_10_100_10_10/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/train_0.txt'
            config.valid_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/valid_0.txt'
            config.test_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/test_0.txt'
    elif config.task_name == 'Tacred':
        config.rel_index = './data/CFRLTacred/rel_index.npy'
        config.relation_name = './data/CFRLTacred/relation_name.txt'
        config.relation_description = './data/CFRLTacred/relation_description_raw.txt'
        if config.num_k == 5:
            config.rel_cluster_label = './data/CFRLTacred/CFRLdata_6_100_5_5/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLTacred/CFRLdata_6_100_5_5/train_0.txt'
            config.valid_data = './data/CFRLTacred/CFRLdata_6_100_5_5/valid_0.txt'
            config.test_data = './data/CFRLTacred/CFRLdata_6_100_5_5/test_0.txt'
        elif config.num_k == 10:
            config.rel_cluster_label = './data/CFRLTacred/CFRLdata_6_100_5_10/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLTacred/CFRLdata_6_100_5_10/train_0.txt'
            config.valid_data = './data/CFRLTacred/CFRLdata_6_100_5_10/valid_0.txt'
            config.test_data = './data/CFRLTacred/CFRLdata_6_100_5_10/test_0.txt'   

    sampler = data_sampler_CFRL(config, seed=42)
    for step, (training_data, valid_data, test_data, current_relations, \
            historic_test_data, seen_relations, seen_descriptions) in enumerate(sampler):
        # print(training_data.keys())
        print(seen_relations)

# dict_keys(['person countries of residence', 'organization top members employees', 'organization member of', 'person origin', 'person title', 'organization country of headquarters'])
# dict_keys(['organization subsidiaries', 'organization parents', 'organization alternate names', 'organization city of headquarters', 'person siblings'])
# dict_keys(['person stateorprovinces of residence', 'person date of death', 'organization number of employees members', 'person alternate names', 'person spouse'])
# dict_keys(['person cities of residence', 'person schools attended', 'person country of death', 'person children', 'person charges'])
# dict_keys(['person country of birth', 'organization website', 'organization shareholders', 'organization dissolved', 'organization founded by'])
# dict_keys(['person date of birth', 'person stateorprovince of birth', 'person parents', 'person employee of', 'person stateorprovince of death'])
# dict_keys(['person cause of death', 'organization political religious affiliation', 'organization stateorprovince of headquarters', 'person other family', 'person city of death'])
# dict_keys(['organization founded', 'person age', 'person city of birth', 'organization members', 'person religion'])