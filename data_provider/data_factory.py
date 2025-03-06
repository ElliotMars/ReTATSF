from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_ReTATSF_weather

def ReTATSF_weather_data_provider(args, flag):
    TS_data_path = args.TS_data_path
    QT_data_path = args.QT_data_path
    NewsDatabase_path = args.NewsDatabase_path

    try:
        assert isinstance(TS_data_path, str)
        assert isinstance(QT_data_path, str)
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

    data_set = Dataset_ReTATSF_weather(
        root_path=args.root_path,
        TS_data_path=TS_data_path,
        QT_data_path=QT_data_path,
        NewsDatabase_path=NewsDatabase_path,
        flag=flag,
        size=[args.seq_len, args.pred_len],
        features=args.features,
        target_id=args.target_id,
        #info_overhead=args.info_overhead,
        #news_pre_embed=args.news_pre_embed,
        #des_pre_embed=args.des_pre_embed,
        #text_encoder=text_encoder,
        #add_date=args.add_date,
        #text_dim=args.text_dim,
        scale=True,
        stride=args.stride  # disable the individual norm
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