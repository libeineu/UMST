# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from fairseq import utils

from . import data_utils, FairseqDataset


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True, src_dict=None,
):
    if len(samples) == 0:
        return {}
    def merge(key, left_pad, move_eos_to_beginning=False):
        if key == "tree":
           return data_utils.collate_tokens(
            [s[key] for s in samples],
            0, eos_idx, left_pad, move_eos_to_beginning,)
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    
    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    tree = merge('tree', True)
    phrase_mapping = merge('phrase_mapping', True)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    #print("src_token:{}".format(src_tokens[0]))
    tree = tree.index_select(0, sort_order)
    #print("tree:{}".format(tree[0]))
    phrase_mapping = phrase_mapping.index_select(0, sort_order)
    #print("phrase_mapping:{}".format(phrase_mapping[0]))
    
    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)
    
    def gen_phrase_matrix(phrase_mapping):
        num_words = torch.max(tree) - 3
        word_length = torch.max(tree, 1)[0] - 3
        diff = (num_words - word_length).numpy()
        mapping =  torch.zeros([phrase_mapping.shape[0], num_words, num_words])
        #print(phrase_mapping.shape)
        #print(mapping.shape)
        size1 = mapping.size(1)
        size2 = mapping.size(2)
        mapping = mapping.numpy()
        #phrase = phrase_mapping[:, :]
        phrase = phrase_mapping
        #if phrase.size(1) %2 != 0:
            #3phrase = phrase_mapping[:, :-1]
            #phrase_mapping = phrase_mapping[:, :-1]

        phrase_mapping = phrase_mapping.tolist()
        #print(phrase_mapping)
        for i in range(phrase.shape[0]):
            for j in range(0, phrase.shape[1], 2):
                if phrase_mapping[i][j] >= 5:
                    if (phrase_mapping[i][j]-5) >= size1 and (phrase_mapping[i][j+1]-5) >=size2:
                        continue
                    else:
                        #if (phrase_mapping[i][j]-5+diff[i]) < size1 or  (phrase_mapping[i][j+1]-5+diff[i]) < size2:
                            #continue
                        #print("j {}".format(j+1))
                        #print(phrase.size(1))
                        #print(j)
                        a = phrase_mapping[i][j]-5+diff[i] if (phrase_mapping[i][j]-5+diff[i]) < size1 else 0
                        b = phrase_mapping[i][j+1]-5+diff[i] if (phrase_mapping[i][j+1]-5+diff[i]) < size2 and (j+1) < phrase.size(1)-1 else 0
                        mapping[i][a][b] = 1
                        mapping[i][b][a] = 1
                        #else:
                        #mapping[i][phrase_mapping[i][j]-5+diff[i]][phrase_mapping[i][j+1]-5+diff[i]] = 1
                        #mapping[i][phrase_mapping[i][j+1]-5+diff[i]][phrase_mapping[i][j]-5+diff[i]] = 1
                else:
                    continue
            
            for j in range(num_words):
                mapping[i][j][j] = 1
            graph = mapping[i]
            deg = np.sum(graph,axis=-1)
            deg = np.diag((deg ** (-0.5)))
            mapping[i] = np.matmul(np.matmul(deg, graph),deg)
            #mapping[i] = deg @ graph @ deg
        mapping = torch.from_numpy(mapping)
            
        return mapping.half().to(tree.device)
        '''
        self_loof = torch.ones(mapping.size(0), mapping.size(1))
        self_loof = torch.diag_embed(self_loof)
        mapping = mapping + self_loof
        deg = torch.sum(mapping,dim=-1)
        deg = torch.diag_embed((deg ** (-0.5)))
        return deg.bmm(mapping).bmm(deg)
        '''
        




    def gen_group_matrix(tree):
        num_words = torch.max(tree) - 3
        num_matrix = num_words - torch.max(tree, 1)[0] + 3
        #print(num_matrix)
        mapping = torch.zeros([tree.shape[0], tree.shape[1], num_words])
        tree_list = tree.tolist()
        for i in range(tree.shape[0]):
            for j in range(tree.shape[1]):
                if tree_list[i][j] >= 5:
                    mapping[i, j, tree_list[i][j]-5+num_matrix[i]] = 1   #  对于大于0部分的都进处理，原来的padding不做处理
                elif tree_list[i][j] ==2:
                    mapping[i, j, -1] =1
        return mapping.half().to(tree.device)
    '''
    def gen_group_matrix_gcn(tree):
        mapping = torch.zeros([tree.shape[0], tree.shape[1], tree.shape[1]])
        tree_list = tree.tolist()
        for i in range(tree.shape[0]):
            for j in range(tree.shape[1]):
                for k in range(tree.shape[1]):
                    if tree_list[i][j] == tree_list[i][k]:
                        mapping[i, j, k] = 1
            graph = mapping[i]
            deg = torch.sum(graph, -1)
            deg = torch.diag((deg ** (-0.5)))
            mapping[i] = deg @ graph @ deg
        return mapping.half().to(tree.device)
    '''
    def gen_group_matrix_gcn(tree:torch.Tensor):
        tree_matrix = tree.unsqueeze(1).eq(tree.unsqueeze(-1)).float()
        #for i in range(tree.shape[0]):
        deg = torch.sum(tree_matrix,dim=-1)
        deg = torch.diag_embed((deg ** (-0.5)))
        return deg.bmm(tree_matrix).bmm(deg)

    def gen_group_matrix_average(tree):
        num_words = torch.max(tree) - 3
        num_matrix = num_words - torch.max(tree, 1)[0] + 3
        #print(num_matrix)
        mapping = torch.zeros([tree.shape[0], tree.shape[1], num_words])
        tree_list = tree.tolist()
        for i in range(tree.shape[0]):
            for j in range(tree.shape[1]):
                if tree_list[i][j] >= 5:
                    mapping[i, j, tree_list[i][j]-5+num_matrix[i]] = 1   #  对于大于0部分的都进处理，原来的padding不做处理
                elif tree_list[i][j] ==2:
                    mapping[i, j, -1] =1
        return mapping.half().to(tree.device)

    def gen_group_matrix_without_scale(tree):
        mapping = torch.zeros([tree.shape[0], tree.shape[1], tree.shape[1]])
        tree_list = tree.tolist()
        for i in range(tree.shape[0]):
            for j in range(tree.shape[1]):
                
                for k in range(tree.shape[1]):
                    if (tree_list[i][j] >= 5 or tree_list[i][j]==2) and tree_list[i][j] == tree_list[i][k]:
                        mapping[i, j, k] = 1
        return mapping.half().to(tree.device)
    
    def gen_group_matrix_weight(tree):
        num_words = torch.max(tree) - 3
        num_matrix = num_words - torch.max(tree, 1)[0] + 3
        #print(num_matrix)
        mapping = torch.zeros([tree.shape[0], tree.shape[1], num_words])
        tree_list = tree.tolist()
        for i in range(tree.shape[0]):
            before = -1
            count = 1
            for j in range(tree.shape[1]):
                if tree_list[i][j] >= 5:
                    if tree_list[i][j] == before:
                        count += 1
                        mapping[i, j, tree_list[i][j]-5+num_matrix[i]] = 1/count   #  对于大于0部分的都进处理，原来的padding不做处理
                    else:
                        mapping[i, j, tree_list[i][j]-5+num_matrix[i]] = 1
                        count = 1
                    before = tree_list[i][j]
                elif tree_list[i][j] ==2:
                    mapping[i, j, -1] =1
        return mapping.half().to(tree.device)
    #a = src_dict.string(src_tokens)
    #print(a)
    #print(tree)
    bpe2word_scale = gen_group_matrix(tree)
    bpe2word_wo_scale = gen_group_matrix_gcn(tree)
    phrase_mapping = gen_phrase_matrix(phrase_mapping)
    #print(phrase_mapping[0])
    #phrase_mapping = bpe2word_scale@phrase@bpe2word_scale.transpose(1,2)
    #phrase_mapping = None   
    #if tree.shape[1]!= src_tokens.shape[1]:
       #for i in range(src_tokens.shape[0]):
           #a = src_dict.string(src_tokens[i])
           #print(a)
           #print(tree[i])
           #print(id[i])
       
	 
    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            "tree": tree,
            'bpe2word_scale': bpe2word_scale,
            'bpe2word_wo_scale': bpe2word_wo_scale,
            'phrase_mapping' : phrase_mapping,
        },
        'target': target,
    }
    #print(batch)
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    #print(batch['ntokens'])
    return batch


class DPTreePhraseGroupLanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for input feeding/teacher forcing
            (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
    """

    def __init__(
        self, src, src_sizes, src_dict, tree, phrase_mapping,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.tree = tree
        self.phrase_mapping = phrase_mapping
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        #a = self.src_dict.string(src_item)
        #print(a)
        tree_item = self.tree[index]
        phrase_mapping = self.phrase_mapping[index]
        #print(tree_item)
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]
        eos = self.src_dict.eos()
        if self.phrase_mapping[index][-1] == eos:
            phrase_mapping = self.phrase_mapping[index][:-1]
        #print(tree_item.shape)
        #print(src_item.shape)
        #tree_item  = tree_item - 4
        #zero = torch.zeros_like(tree_item)
        #tree_item = torch.where(tree_item>0, tree_item, zero)
        def gen_group_matrix(tree):
            num_words = torch.max(tree)
            mapping = torch.zeros([tree.shape[0], num_words])
            tree_list = tree.tolist()
            for j in range(tree.shape[0]):
                if tree[j] != 0:
                   mapping[j, tree[j]-1] = 1 
            return mapping.half().to(tree.device)
        return {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'tree': tree_item,
            'phrase_mapping':phrase_mapping,
            #"mapping": gen_group_matrix(tree_item)
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,src_dict=self.src_dict
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        src_len, tgt_len = utils.resolve_max_positions(
            (src_len, tgt_len),
            max_positions,
            (self.max_source_positions, self.max_target_positions),
        )
        #print(1)
        bsz = max(num_tokens // max(src_len, tgt_len), 1)
        return self.collater([
            {
                'id': i,
                'source': self.src_dict.dummy_sentence(src_len),
                'target': self.tgt_dict.dummy_sentence(tgt_len) if self.tgt_dict is not None else None,
                'tree': torch.Tensor([i] for i in range(5, 5+src_len)).half(),
                "mapping": torch.Tensor([0] * ( src_len)).half(),
                "phrase_mapping":torch.Tensor([6] * (src_len * 2)).long(),
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and getattr(self.tgt, 'supports_prefetch', False)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        self.tgt.prefetch(indices)
