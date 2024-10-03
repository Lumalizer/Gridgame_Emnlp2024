# inspired by egg.zoo.signal_game.archs
import torch
import torch.nn as nn
import torch.nn.functional as F
from options import Options
from experiment.models.graph_embeddings import GraphPositionNodeEmbeddings, GraphEmbeddings
from experiment.models.image_embeddings import ImageEmbeddings
from typing import Union


class BaseSender(nn.Module):
    def __init__(self, options: Options, embedder_network: Union[ImageEmbeddings, GraphEmbeddings, GraphPositionNodeEmbeddings]):
        super(BaseSender, self).__init__()
        self.options = options
        self.view_size = 1

        self.embedder = embedder_network(options)

        self.conv1 = nn.Conv2d(1, options.hidden_size, kernel_size=(
            self.view_size, 1), stride=(self.view_size, 1), bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(options.hidden_size, 1), stride=(options.hidden_size, 1), bias=False)
        self.lin1 = nn.Linear(options.embedding_size, options.hidden_size, bias=False)

        self.bn1 = nn.BatchNorm2d(options.hidden_size)
        self.bn2 = nn.BatchNorm2d(1)

        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()

        self.lin2 = nn.Linear(options.hidden_size, 50, bias=False)
        self.lin3 = nn.Linear(options.hidden_size, 50, bias=False)
        self.lin4 = nn.Linear(options.hidden_size, 4, bias=False)
        self.lin5 = nn.Linear(options.hidden_size, 4, bias=False)

    def forward(self, x: torch.Tensor, _aux_input):
        x = _aux_input['data_sender']

        for p in self.embedder.parameters():
            p.requires_grad = True

        x = self.embedder(x)
        x = x.view(self.options.batch_size, self.view_size, -1)

        emb = x.unsqueeze(dim=1)                # batch_size x 1 x game_size x embedding_size
        h = self.conv1(emb)                     # batch_size x hidden_size x 1 x embedding_size
        h = self.bn1(h)
        h = self.tanh1(h)

        h = h.transpose(1, 2)                   # batch_size, 1, hidden_size, embedding_size
        h = self.conv2(h)                       # batch_size, 1, 1, embedding_size
        h = self.bn2(h)
        h = self.tanh2(h)

        h = h.squeeze()                         # batch_size x embedding_size
        h = self.lin1(h)                        # batch_size x hidden_size

        # if not self.options._eval:
        #     _aux_input['vectors_sender'] = h

        if self.options.use_shape_subtasks:
            _aux_input['shape1_pred'] = F.log_softmax(self.lin2(h), dim=1)
            _aux_input['shape2_pred'] = F.log_softmax(self.lin3(h), dim=1)
        if self.options.use_position_subtasks:
            _aux_input['pos1_pred'] = F.log_softmax(self.lin4(h), dim=1)
            _aux_input['pos2_pred'] = F.log_softmax(self.lin5(h), dim=1)
        return h


class BaseReceiver(nn.Module):
    def __init__(self, options: Options, embedder_network: Union[ImageEmbeddings, GraphEmbeddings, GraphPositionNodeEmbeddings]):
        super(BaseReceiver, self).__init__()
        self.options = options
        self.lin1 = nn.Linear(options.hidden_size, options.embedding_size)

        self.embedder = embedder_network(options)

    def forward(self, signal, x, _aux_input):
        x = _aux_input['data_receiver']

        x = self.embedder(x)
        x = x.view(self.options.batch_size, self.options.game_size, -1)

        h_s = self.lin1(signal)                 # embed the signal
        h_s = h_s.unsqueeze(dim=1)              # batch_size x embedding_size
        h_s = h_s.transpose(1, 2)               # batch_size x 1 x embedding_size

        out = torch.bmm(x, h_s)                 # batch_size x embedding_size x 1
        out = out.squeeze(dim=-1)               # batch_size x game_size x 1

        _aux_input['receiver_probs'] = F.softmax(out, dim=1).round(decimals=2)  # log receiver probabilities
        log_probs = F.log_softmax(out, dim=1)   # batch_size x game_size
        return log_probs


class ImageSender(BaseSender):
    def __init__(self, options: Options):
        super().__init__(options, ImageEmbeddings)


class ImageReceiver(BaseReceiver):
    def __init__(self, options: Options):
        super().__init__(options, ImageEmbeddings)


class GraphSender(BaseSender):
    def __init__(self, options: Options):
        super().__init__(
            options,
            GraphEmbeddings if options.node_encoding == 'edgeattr' else GraphPositionNodeEmbeddings
        )


class GraphReceiver(BaseReceiver):
    def __init__(self, options: Options):
        super().__init__(
            options,
            GraphEmbeddings if options.node_encoding == 'edgeattr' else GraphPositionNodeEmbeddings
        )
