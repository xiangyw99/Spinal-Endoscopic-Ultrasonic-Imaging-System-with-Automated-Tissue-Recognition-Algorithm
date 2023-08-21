class CFG:


    output_dir  = './logs/'

    # model and save model
    experiment  = 'resnet50'
    model       = 'resnet34'
    pretrained  = True

    # training
    epochs      = 150

    lr = 0.0003
    min_lr        = 1e-6
    batch_size    = 32
    T_max         = int(30000/batch_size*epochs)+50
    T_0           = 10
    warmup_epochs = 10
    # test

    scheduler     = 'CosineAnnealingLR'
    n_accumulate  = 4

    # data prepare
    imagesTr  = '/home/data/xiangyw/spine_image_0529_strict_split/'
    model_path= '/home/xiangyw/spine_classification_github/checkpoints/'

    num_class        = 6
    k_folds          = 5
    nw = 8  # num_workers

