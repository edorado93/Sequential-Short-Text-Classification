class Config:
    eval_using = "loss"
    relative_train_path = '/data/85%/arxiv/train.dat'
    relative_dev_path = '/data/85%/arxiv/valid.dat'
    min_freq = 1
    emsize = 512
    hidden_size = 1150
    batch_size = 8
    lr = 0.00001
    log_interval = 200
    max_grad_norm = 10
    d1 = 0
    d2 = 0
    epochs = 100
    patience = 20
    input_dropout = 0
    pretrained = None
    weight_decay = 0
    bidirectional = True

    # Attention based scoring configurations
    use_attention = False
    attention_score_type = "softmax"
    attention_type = "weighted"

class D2Config_1(Config):
    d2 = 1

class D2_1_L2(Config):
    d2 = 1
    weight_decay = 0.9

class D2_1_L2_Sigmoid_Weighted(D2_1_L2):
    d2 = 1
    use_attention = True
    attention_score_type = "sigmoid"
    attention_type = "weighted"

class D2_1_L2_Sigmoid_Concat(D2_1_L2):
    d2 = 1
    use_attention = True
    attention_score_type = "sigmoid"
    attention_type = "concat"

class D2_1_L2_Softmax_Weighted(D2_1_L2):
    d2 = 1
    use_attention = True
    attention_score_type = "softmax"
    attention_type = "weighted"

class D2_1_L2_Softmax_Concat(D2_1_L2):
    d2 = 1
    use_attention = True
    attention_score_type = "softmax"
    attention_type = "concat"

def get_conf(conf):
    return config[conf]

config = {"d2_1" : D2_1_L2(),
          "sig_weight": D2_1_L2_Sigmoid_Weighted(),
          "sig_concat": D2_1_L2_Sigmoid_Concat(),
          "softmax_weight": D2_1_L2_Softmax_Weighted(),
          "softmax_concat": D2_1_L2_Softmax_Concat()}
