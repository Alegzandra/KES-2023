import fasttext
import fasttext.util
import os


#fasttext.util.download_model('ro', if_exists='ignore')

#model = fasttext.train_supervised('fasttext/train_val.txt', lr = 0.001, dim = 5, ws = 5, epoch = 1000, pretrainedVectors = 'cc.ro.300.bin')
#model = fasttext.train_supervised('fasttext/train_val.txt', lr = 0.001, minn=2, maxn=3, dim = 300, ws = 2, wordNgrams = 2, epoch = 1000, pretrainedVectors = 'cc.ro.300.vec')
model = fasttext.train_supervised(input='train.txt', autotuneValidationFile='val.txt', autotuneDuration=600)
#model = fasttext.train_supervised('fasttext/train_val.txt', lr = 0.0001, minn=2, maxn=3, ws = 2, wordNgrams = 2, epoch = 10000)

#   ['input', 'lr', 'dim', 'ws', 'epoch', 'minCount',
# 'minCountLabel', 'minn', 'maxn', 'neg', 'wordNgrams', 'loss', 'bucket',
# 'thread', 'lrUpdateRate', 't', 'label', 'verbose', 'pretrainedVectors',
# 'seed', 'autotuneValidationFile', 'autotuneMetric',
# 'autotunePredictions', 'autotuneDuration', 'autotuneModelSize']


# autotuneDuration=600
# lr                # learning rate [0.1]
# dim               # size of word vectors [100]
# ws                # size of the context window [5]
# epoch             # number of epochs [5]
# minCount          # minimal number of word occurences [1]
# minCountLabel     # minimal number of label occurences [1]
# minn              # min length of char ngram [0]
# maxn              # max length of char ngram [0]
# neg               # number of negatives sampled [5]
# wordNgrams        # max length of word ngram [1]
# loss              # loss function {ns, hs, softmax, ova} [softmax]
# bucket            # number of buckets [2000000]
# thread            # number of threads [number of cpus]
# lrUpdateRate      # change the rate of updates for the learning rate [100]
# t                 # sampling threshold [0.0001]
# label             # label prefix ['__label__']
# verbose           # verbose [2]
# pretrainedVectors # pretrained word vectors (.vec file) for supervised learning []

model.save_model("model1.bin")
