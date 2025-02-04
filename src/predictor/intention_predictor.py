import torch
import torch.nn as nn
from collections import namedtuple

from .rnn_model import RNNParameters, RNNEncoder, LinearParameters, LinearModel, AttentionParameters


class IntentionPredictor(nn.Module):
    def __init__(
            self, 
            hparams_rnn_hist : RNNParameters, 
            hparams_rnn_plan : RNNParameters, 
            hparams_attn_hist : AttentionParameters,
            hparams_attn_plan : AttentionParameters,
            hparams_linear : LinearParameters,
            hparams_goal_hist : RNNParameters = None
        ):
        super(IntentionPredictor, self).__init__()

        # RNN model for human and robot trajectory history
        self.hist_encoder = self._create_encoder(hparams_rnn_hist)
        # RNN model for robot's future plan
        self.plan_encoder = self._create_encoder(hparams_rnn_plan)
        # attention layer for history encoder
        self.hist_attn = self._create_attention(hparams_attn_hist)
        # attention layer for robot's future plan
        self.plan_attn = self._create_attention(hparams_attn_plan)
        # linear layers for classifier
        self.classifier = self._create_classifier(hparams_linear)
        if hparams_goal_hist is not None:
            self.goal_hist_encoder = self._create_encoder(hparams_goal_hist)
            self.goal_hist = True
        else:
            self.goal_hist = False

    def _create_encoder(self, hparams):
        encoder = RNNEncoder(**hparams)
        return encoder

    def _create_attention(self, hparams):
        attention = nn.MultiheadAttention(**hparams)
        return attention

    def _create_classifier(self, hparams):
        linear = LinearModel(**hparams)
        return linear

    def forward(self, seq_hist, r_plan, goals):
        """
        sec : state sequence
        goals : the possible goals of the human
        """
        
        # TODO: check for gradients through RNN
        # run through sequences RNN encoder
        hist_enc_out = self.hist_encoder(seq_hist)[0]
        plan_enc_out = self.plan_encoder(r_plan)[0]

        # compute attention
        # TODO: torch 1.7.0 doesn't support batch_first for attention layer
        # hist_enc_out, _ = self.hist_attn(hist_enc_out, hist_enc_out, hist_enc_out)
        # plan_enc_out, _ = self.plan_attn(plan_enc_out, plan_enc_out, plan_enc_out)

        # flatten and concat outputs
        hist_enc_out = torch.flatten(hist_enc_out, start_dim=1)
        plan_enc_out = torch.flatten(plan_enc_out, start_dim=1)
        if self.goal_hist:
            goal_hist_enc_out = self.goal_hist_encoder(goals)[0]
            goal_hist_enc_out = torch.flatten(goal_hist_enc_out, start_dim=1)
            c_in = torch.cat((hist_enc_out, plan_enc_out, goal_hist_enc_out), dim=1)
        else:
            goals = torch.flatten(goals, start_dim=1)
            c_in = torch.cat((hist_enc_out, plan_enc_out, goals), dim=1)

        return self.classifier(c_in)

def create_model(horizon_len=5, goal_mode : str = "static"):
    # TODO: figure out where this should be stored
    seq_len = 5
    state_dim = 4
    n_goals = 3

    # create RNNParameters object
    rnn_hist_params = RNNParameters(
                    feat_dim=8,
                    num_layers=2,
                    hidden_size=128,
                    droupout_fc=0.0
                    )

    rnn_plan_params = RNNParameters(
                    feat_dim=4,
                    num_layers=2,
                    hidden_size=128,
                    droupout_fc=0.0
                    )

    # create AttentionParameters object
    attn_params = AttentionParameters(
                    embed_dim=128,
                    num_heads=1,
                    # batch_first=True
                    )

    # create LinearParameters object
    if goal_mode == "dynamic":
        goal_hist_params = RNNParameters(
                        feat_dim=12,
                        num_layers=2,
                        hidden_size=128,
                        droupout_fc=0.0
                        )
        fc_params = LinearParameters(
                    in_dim=128*seq_len*2 + (128*horizon_len),
                    out_dim=3,
                    n_hidden=2,
                    hidden_dim=128
                    )
    elif goal_mode == "static":
        goal_hist_params = None
        fc_params = LinearParameters(
                        in_dim=(128*seq_len + (128*horizon_len) + state_dim*n_goals),
                        out_dim=3,
                        n_hidden=2,
                        hidden_dim=128
                        )

    model = IntentionPredictor(rnn_hist_params, rnn_plan_params, attn_params, attn_params, fc_params, goal_hist_params)

    return model