from options import Options
import torch.nn.functional as F


def loss_nll(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input, options: Options):
    nll = F.nll_loss(receiver_output, labels, reduction="none")
    acc = (labels == receiver_output.argmax(dim=1)).float()
    acc = {"acc": acc}

    if options.use_shape_subtasks:
        nll += 0.2 * F.nll_loss(_aux_input['shape1_pred'], _aux_input['shape1'], reduction="none")
        nll += 0.2 * F.nll_loss(_aux_input['shape2_pred'], _aux_input['shape2'], reduction="none")

        acc['shape1acc'] = (_aux_input['shape1'] == _aux_input['shape1_pred'].argmax(dim=1)).float()
        acc['shape2acc'] = (_aux_input['shape2'] == _aux_input['shape2_pred'].argmax(dim=1)).float()

    if options.use_position_subtasks:
        nll += 0.2 * F.nll_loss(_aux_input['pos1_pred'], _aux_input['pos1'], reduction="none")
        nll += 0.2 * F.nll_loss(_aux_input['pos2_pred'], _aux_input['pos2'], reduction="none")

        acc['pos1acc'] = (_aux_input['pos1'] == _aux_input['pos1_pred'].argmax(dim=1)).float()
        acc['pos2acc'] = (_aux_input['pos2'] == _aux_input['pos2_pred'].argmax(dim=1)).float()

    return nll, acc


def loss_crossentropy(
    _sender_input,
    _message,
    _receiver_input,
    receiver_output,
    labels,
    _aux_input,
):
    # to avoid the case when all the logits are equal
    score = receiver_output.detach()
    score[:, 0] -= 1e-10  # we assume that the first is the correct one
    acc = (score.argmax(dim=1) == labels).float()

    loss = F.cross_entropy(receiver_output, labels, reduction="none")
    # positive = score[:, :1].repeat(1, game_size-1)
    # negative = score[:, 1:]
    # loss = (negative + 0.1 - positive).max(-1)[0]

    return loss, {"acc": acc}
