import torch    


def compute_invariant_position(query_length, key_length, type_ids, row_ids, col_ids):
        """ Compute binned relative position bias for table"""
        # assume query_length == key_length

        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position_template = memory_position - context_position  # shape (query_length, key_length)
        # shape of relative_position_template (batch_size, query_length, key_length)
        relative_position_template = relative_position_template.unsqueeze(0).repeat(type_ids.shape[0], 1, 1).to(
            type_ids.device)

        # relative position for meta data
        # others are set to 0
        meta_relative_position = relative_position_template.clone()
        meta_mask = torch.logical_and(type_ids < 2.5, type_ids > 0.5)  # shape (batch_size, query_length)
        # shape of meta_mask (batch_size, query_length, key_length)
        meta_mask = torch.bmm(torch.unsqueeze(meta_mask.float(), 2), torch.unsqueeze(meta_mask.float(), 1)) > 0.5
        meta_relative_position = meta_relative_position * meta_mask

        # relative position for cells
        # others are set to 0
        cell_relative_position = relative_position_template.clone()
        cell_mask = type_ids == 3  # shape (batch_size, query_length)
        # shape of cell_mask (batch_size, query_length, key_length)
        cell_mask = torch.bmm(torch.unsqueeze(cell_mask.float(), 2), torch.unsqueeze(cell_mask.float(), 1)) > 0.5

        row_diff = torch.abs(row_ids.unsqueeze(-1) - row_ids.unsqueeze(1))  # shape (batch_size, query_length, key_length)
        col_diff = torch.abs(col_ids.unsqueeze(-1) - col_ids.unsqueeze(1))  # shape (batch_size, query_length, key_length)

        same_cell_mask = torch.logical_and(row_diff + col_diff < 0.5, cell_mask)

        rc_cell_mask = torch.logical_and(torch.logical_not(same_cell_mask), cell_mask)

        cell_relative_position = cell_relative_position * same_cell_mask + 512 * rc_cell_mask

        # relative position between meta data and cell
        bridge_mask = torch.logical_not(meta_mask + cell_mask)  # 1 for attention between meta data and cell
        bridge_relative_position = 512 * bridge_mask

        # For a table:
        # A B
        # C D
        # where A for metadata, D for cells, B and C for attention between metadata and cells
        relative_position = meta_relative_position + cell_relative_position + bridge_relative_position

        return relative_position
