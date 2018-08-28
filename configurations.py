class Config:
    eval_using = "accuracy"
    relative_train_path = '/data/arxiv/train.dat'
    relative_dev_path = '/data/arxiv/valid.dat'
    min_freq = 1
    emsize = 512
    hidden_size = 1150
    batch_size = 8
    lr = 0.00001
    log_interval = 200
    max_grad_norm = 10
    d1 = 0
    d2 = 0
    epochs = 40
    patience = 12
    input_dropout = 0
    pretrained = None
    weight_decay = 0
    bidirectional = True

class D1Config_1(Config):
    d1 = 1

class D1Config_2(Config):
    d1 = 2

class D2Config_1(Config):
    d2 = 1

class D2Config_1WeightDecay(Config):
    d2 = 1

class D2Config_1Dropout(Config):
    d2 = 1
    input_dropout = 0.2

class D2Config_2(Config):
    d2 = 2

class D1D2Config_1(Config):
    d1 = 1
    d2 = 1

class D1D2Config_2(Config):
    d1 = 2
    d2 = 2

class D1D2Dropout(D1D2Config_1):
    input_dropout = 0.2

class D1D2Pretrained(D1D2Config_1):
    pretrained = 'embeddings/arxiv/embedding.vec'

class D1D2PretrainedDropout(D1D2Config_1):
    pretrained = 'embeddings/arxiv/embedding.vec'
    input_dropout = 0.2

def get_conf(conf):
    return config[conf]

config = {"d1_1" : D1Config_1(),
          "d1_2" : D1Config_2(),
          "d2_1" : D2Config_1(),
          "d2_1_drop": D2Config_1Dropout(),
          "d2_1_L2": D2Config_1WeightDecay(),
          "d2_2" : D2Config_2(),
          "d1d2_1" : D1D2Config_1(),
          "d1d2_2" : D1D2Config_2(),
          "d1d2drop" : D1D2Dropout(),
          "d1d2pre" : D1D2Pretrained(),
          "d1d2predrop" : D1D2PretrainedDropout()}
