import torch
from transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F
from configs import parse_arguments
from torch.nn.utils.rnn import unpad_sequence
from random import shuffle

args = parse_arguments()
device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")  # type: ignore


class BertED(nn.Module):
    def __init__(self, class_num=args.class_num + 1, input_map=False):
        super().__init__()
        self.backbone = BertModel.from_pretrained(args.backbone)
        if not args.no_freeze_bert:
            print("Freeze bert parameters")
            for _, param in list(self.backbone.named_parameters()):
                param.requires_grad = False
        else:
            print("Update bert parameters")
        self.is_input_mapping = input_map
        self.input_dim = self.backbone.config.hidden_size
        self.fc = nn.Linear(self.input_dim, class_num)
        self.b_fc = nn.Linear(self.input_dim, 1)
        if self.is_input_mapping:
            self.map_hidden_dim = 512 # 512 is implemented by the paper
            self.map_input_dim =  self.input_dim * 2
            self.input_map = nn.Sequential(
                nn.Linear(self.map_input_dim, self.map_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.map_hidden_dim, self.map_hidden_dim),
                nn.ReLU(),
            )
            self.fc = nn.Linear(self.map_hidden_dim, class_num)

    def forward(self, x, masks, span=None, aug=None, top_k=None):
        # x = self.backbone(x) #TODO: test use
        return_dict = {}
        backbone_output = self.backbone(x, attention_mask = masks, output_attentions=True)
        x, pooled_feat, attention_scores = backbone_output[0], backbone_output[1], backbone_output[2] # x [B, L, H], pooled_feat [B, H]

        if top_k is not None:
            # attentions_layer: (B, H, S, S)
            attentions_layer = attention_scores[-1]  # choose layer (e.g., last layer)

            # Column-wise sum over queries (dim=2). Result: (B, H, S)
            col_sum_per_head = attentions_layer.sum(dim=2)

            # Aggregate across heads -> (B, S)
            token_scores = col_sum_per_head.mean(dim=1)  # (B, S)

            # Mask out padding tokens (if you have attention_mask of shape (B, S))
            token_scores = token_scores * masks.to(token_scores.dtype)  # zero-out pad positions

            # Normalize to probabilities per example
            token_probs = token_scores / (token_scores.sum(dim=1, keepdim=True) + 1e-12)

            # Choose safe k (don't request more than seq length)
            S = token_probs.size(1)
            k = top_k if S > top_k else S

            # top-k indices and scores per example
            topk_scores, topk_indices = torch.topk(token_probs, k=k, dim=1)

            B, _, H = x.size()

            # 8) gather hidden vectors for top-k indices -> (B, k, H)
            idx_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, H)   # (B, k, H)
            topk_context_feature = torch.gather(x, dim=1, index=idx_expanded)  # (B, k, H)
        else:
            topk_context_feature = None
        return_dict['topk_context_feature'] = topk_context_feature
        

        context_feature = x.view(-1, x.shape[-1]) # context_feature [B*L, H]
        return_dict['reps'] = x[:, 0, :].clone()

        if span != None:
            outputs, trig_feature = [], []
            for i in range(len(span)):
                if self.is_input_mapping:
                    x_cdt = torch.stack([torch.index_select(x[i], 0, span[i][:, j]) for j in range(span[i].size(-1))])
                    x_cdt = x_cdt.permute(1, 0, 2)
                    x_cdt = x_cdt.contiguous().view(x_cdt.size(0), x_cdt.size(-1) * 2)
                    opt = self.input_map(x_cdt)
                else:
                    opt = torch.index_select(x[i], 0, span[i][:, 0]) + torch.index_select(x[i], 0, span[i][:, 1])
                    # x = x_cdt.permute(1, 0, 2) 
                trig_feature.append(opt)
            trig_feature = torch.cat(trig_feature)
        
        outputs = self.fc(trig_feature)
        return_dict['outputs'] = outputs
        return_dict['context_feat'] = context_feature
        return_dict['trig_feat'] = trig_feature
        return_dict['binary_logits'] = F.sigmoid(self.b_fc(trig_feature))
        # if args.single_label:
        #     return_outputs = self.fc(enc_out_feature).view(-1, args.class_num + 1)
        # else:
        #     return_outputs = self.fc(feature)
        if aug is not None:
            feature_aug = trig_feature + torch.randn_like(trig_feature) * aug
            outputs_aug = self.fc(feature_aug)
            return_dict['feature_aug'] = feature_aug
            return_dict['outputs_aug'] = outputs_aug
        return return_dict

    def forward_backbone(self, x, masks):
        x = self.backbone(x, attention_mask = masks)
        x = x.last_hidden_state
        return x

    def forward_input_map(self, x):
        return self.input_map(x)
