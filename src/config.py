def config(args):
    '''
    Configure details for each dataset.
    '''
    if 'FB' in args.data_dir:
        args.bandwidth = 400
        args.num_epochs = 100
        args.num_wait_epochs = 50
        args.num_peek_epochs = 2
        args.batch_size = 256  # 128
        args.train_batch_size = 256  # 128
        args.dev_batch_size = 4  # 2
        args.grad_norm = 0
        args.emb_dropout_rate = 0.3
        args.ff_dropout_rate = 0.1
        args.action_dropout_rate = 0.5
        args.beta = 0.02
        args.beam_size = 128
        # ARGCN
        args.neigh_dropout = 0.3
        args.node_dropout = 0.3
        # Augmentation link
        args.aug_link_threshold = 0.1
        args.aug_link_support_threshold = 1.0
    elif 'WN' in args.data_dir:
        args.bandwidth = 256
        args.num_epochs = 50 # 20
        args.num_wait_epochs = 50 # 20
        args.num_peek_epochs = 1
        args.batch_size = 1024
        args.train_batch_size = 1024
        args.dev_batch_size = 128
        args.grad_norm = 0
        args.emb_dropout_rate = 0.5
        args.ff_dropout_rate = 0.3
        args.action_dropout_rate = 0.5
        args.beta = 0
        args.beam_size = 128
        # ARGCN
        args.neigh_dropout = 0.7
        args.node_dropout = 0.7
        # Augmentation Link
        args.aug_link_threshold = 0.3
        args.aug_link_support_threshold = 1.0
    elif 'NELL' in args.data_dir:
        args.bandwidth = 400
        args.num_epochs = 1000
        args.num_wait_epochs = 100
        args.num_peek_epochs = 1
        args.bucket_interval = 10
        args.batch_size = 256
        args.train_batch_size = 256
        args.dev_batch_size = 32
        args.grad_norm = 5
        args.emb_dropout_rate = 0.3
        args.ff_dropout_rate = 0.1
        args.action_dropout_rate = 0.0
        args.beta = 0.05
        args.beam_size = 128
        # ARGCN
        args.neigh_dropout = 0.3
        args.node_dropout = 0.7
        # Augmentation link
        args.aug_link_threshold = 0.1
        args.aug_link_support_threshold = 1.0