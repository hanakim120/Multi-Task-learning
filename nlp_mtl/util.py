import torch
from torch.autograd import Variable


def get_batch(source, *targets, batch_size, seq_len=10, cuda=False, evalu=False):
    """Generate batch from the raw data."""
    # source.size() : torch.Size([4000])
    nbatch = source.size(0) // batch_size  # source.size(0) : dim 0

    shuffle_mask = torch.randperm(batch_size)
    # Trim extra elements doesn't fit well
    source = source.narrow(0, 0, nbatch * batch_size)  # torch.narrow(input, dim, start, length)

    # Make batch shape
    source = source.view(batch_size, -1).t().contiguous()

    # Shuffle (shuffle in batches)
    source = source[:, shuffle_mask]

    if cuda:
        source = source.cuda()
    #print(targets,'############targets############')
    targets = list(targets)
    for i in range(len(targets)):
        targets[i] = targets[i].narrow(0, 0, nbatch * batch_size)
        targets[i] = targets[i].view(batch_size, -1).t().contiguous()
        targets[i] = targets[i][:, shuffle_mask]
        if cuda:
            targets[i] = targets[i].cuda()

    for i in range(source.size(0) // seq_len):
        ys = []
        X = Variable(source[i * seq_len:(i + 1) * seq_len], volatile=evalu)
        for target in targets:
            ys.append(Variable(target[i * seq_len:(i + 1) * seq_len]))
        # print(X, '#######XXXXXXXXX##########')
        #print(ys, "$$$$$$$$$ys$$$$$$$$$$$$$$$$$")
        yield X, ys

def repackage_hidden(h):
    """Wrap hidden in the new Variable to detach it from old history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)
