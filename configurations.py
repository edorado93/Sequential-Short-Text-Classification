class Config:
    eval_using = "accuracy"
    relative_train_path = '/data/train.dat'
    relative_dev_path = '/data/valid.dat'
    relative_test_path = '/data/valid.dat'
    min_freq = 1
    emsize = 512
    hidden_size = emsize
    batch_size = 8
    lr = 0.001
    log_interval = 200
    max_grad_norm = 10
    d1 = 0
    d2 = 0
    epochs = 40
    patience = 12
    pooling_dropout = 0
    pretrained = None

class D1Config_1(Config):
    d1 = 1

class D1Config_2(Config):
    d1 = 2

class D2Config_1(Config):
    d2 = 1

class D2Config_2(Config):
    d2 = 2

class D1D2Config_1(Config):
    d1 = 1
    d2 = 1

class D1D2Config_2(Config):
    d1 = 2
    d2 = 2

class D1D2Dropout(D1D2Config_1):
    pooling_dropout = 0.2

class D1D2Pretrained(D1D2Config_1):
    pretrained = 'embeddings/embedding.vec'

class D1D2PretrainedDropout(D1D2Config_1):
    pretrained = 'embeddings/embedding.vec'
    pooling_dropout = 0.2

def get_conf(conf):
    return config[conf]

config = {"d1_1" : D1Config_1(),
          "d1_2" : D1Config_2(),
          "d2_1" : D2Config_1(),
          "d2_2" : D2Config_2(),
          "d1d2_1" : D1D2Config_1(),
          "d1d2_2" : D1D2Config_2(),
          "d1d2drop" : D1D2Dropout(),
          "d1d2pre" : D1D2Pretrained(),
          "d1d2predrop" : D1D2PretrainedDropout()}
