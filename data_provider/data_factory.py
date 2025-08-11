from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_ReTATSF_Energy
import torch

def ReTATSF_Energy_data_provider(args, flag, target_ids):
    TS_data_path = args.TS_data_path
    QT_emb_path = args.QT_emb_path
    Des_emb_path = args.Des_emb_path
    NewsDatabase_path = args.NewsDatabase_path

    try:
        assert isinstance(TS_data_path, str)
        assert isinstance(NewsDatabase_path, str)
    except AssertionError:
        print('Data path should be a string!')
        exit()

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size

    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size

    data_set = Dataset_ReTATSF_Energy(
        root_path=args.root_path,
        TS_data_path=TS_data_path,
        QT_emb_path=QT_emb_path,
        Des_emb_path=Des_emb_path,
        NewsDatabase_path=NewsDatabase_path,
        flag=flag,
        size=[args.seq_len, args.pred_len, args.label_len],
        target_ids=target_ids,
        scale=True,
        stride=args.stride,
        num_data=args.num_data# disable the individual norm
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=custom_collate_fn)
    return data_set, data_loader

def custom_collate_fn(batch):
    batch_target_series_x = torch.stack([torch.tensor(item[0]).permute(1, 0) for item in batch])
    batch_target_series_y = torch.stack([torch.tensor(item[1]).permute(1, 0) for item in batch])
    batch_TS_database = torch.stack([torch.tensor(item[2]).permute(1, 0) for item in batch])
    batch_qt = torch.stack([torch.tensor(item[3]) for item in batch])
    batch_des = torch.stack([torch.tensor(item[4]).unsqueeze(1) for item in batch])
    batch_newsdatabase = torch.stack([torch.tensor(item[5]) for item in batch])

    return batch_target_series_x, batch_target_series_y, batch_TS_database, batch_qt, batch_des, batch_newsdatabase