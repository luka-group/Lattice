import torch


def rewrite_encoder_attention_mask(general_attention_mask, type_ids, row_ids, col_ids,
                                   type_edges=((1, 1), (2, 2), (1, 2), (2, 1), (3, 1), (3, 2), (1, 3), (2, 3))):
    """
    :param general_attention_mask: 1 for tokens of type 1-3, 0 for padding
    :param type_ids: 1 for page title, 2 for section title, 3 for cells 
    :param row_ids: the index of row, starting from 1, 0 for metadata
    :param col_ids: the index of column, starting from 1, 0 for metadata
    :param type_edges: additional edges between types, the information propagate according to (from_type, to_type)
    :return: attention_mask [batch_size, query_length, key_length]
    """
    # mask by padding
    general_attention_masks = torch.bmm(torch.unsqueeze(general_attention_mask.float(), 2),
                                        torch.unsqueeze(general_attention_mask.float(), 1)) > 0.5

    # mask by type (assign 1 for !c2c attention)
    type_masks = torch.zeros_like(general_attention_masks)
    for key_type, query_type in type_edges:
        # information propagates from key to query
        query_type_mask = type_ids == query_type
        key_type_mask = type_ids == key_type
        edge_mask = torch.bmm(torch.unsqueeze(query_type_mask.float(), 2),
                              torch.unsqueeze(key_type_mask.float(), 1)) > 0.5
        type_masks = torch.logical_or(type_masks, edge_mask)

    c2c_mask = torch.logical_and(general_attention_masks, torch.logical_not(type_masks))

    # mask by row and col
    row_diff = torch.abs(torch.unsqueeze(row_ids, 2) - torch.unsqueeze(row_ids, 1))
    col_diff = torch.abs(torch.unsqueeze(col_ids, 2) - torch.unsqueeze(col_ids, 1))

    # same row or/and column, and should be cell token
    adjacent_masks = torch.logical_and(torch.logical_or(row_diff < 0.5, col_diff < 0.5), c2c_mask)

    # final mask
    attention_masks = torch.logical_and(general_attention_masks, torch.logical_or(type_masks, adjacent_masks))

    return attention_masks
