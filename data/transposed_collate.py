from torch.utils.data.dataloader import default_collate


def transposed_collate(batch):
    """
    Wrapper around the default collate function to return sequences of PyTorch
    tensors with sequence step as the first dimension and batch index as the
    second dimension.

    Args:
        batch (list): data examples
    """
    collated_batch = default_collate(batch)
    transposed_batch = collated_batch.transpose_(0, 1)
    return transposed_batch
