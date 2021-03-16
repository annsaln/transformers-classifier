#!/usr/bin/env python3

import sys
import math
import numpy as np
from os.path import isfile
import csv
import json

from scipy.sparse import lil_matrix

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dropout, Dense, TimeDistributed, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, AUC
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.config import experimental as tfconf_exp

from tensorflow_addons.metrics import F1Score

from transformers import AutoConfig, AutoTokenizer, TFAutoModel, TFAutoModelForSequenceClassification
from transformers.optimization_tf import create_optimizer

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from logging import warning

from readers import READERS, get_reader
from common import timed


# Parameter defaults
DEFAULT_BATCH_SIZE = 8
DEFAULT_SEQ_LEN = 512
DEFAULT_LR = 5e-5
DEFAULT_WARMUP_PROPORTION = 0.1


def init_tf_memory():
    gpus = tfconf_exp.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tfconf_exp.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model_name', default=None,
                    help='pretrained model name')
    ap.add_argument('--train', metavar='FILE', required=True,
                    help='training data')
    ap.add_argument('--dev', metavar='FILE', required=True,
                    help='development data')
    ap.add_argument('--test', metavar='FILE', required=False,
                    help='test data', default=None)
    ap.add_argument('--batch_size', metavar='INT', type=int,
                    default=DEFAULT_BATCH_SIZE,
                    help='batch size for training')
    ap.add_argument('--epochs', metavar='INT', type=int, default=1,
                    help='number of training epochs')
    ap.add_argument('--lr', '--learning_rate', metavar='FLOAT', type=float,
                    default=DEFAULT_LR, help='learning rate')
    ap.add_argument('--seq_len', metavar='INT', type=int,
                    default=DEFAULT_SEQ_LEN,
                    help='maximum input sequence length')
    ap.add_argument('--warmup_proportion', metavar='FLOAT', type=float,
                    default=DEFAULT_WARMUP_PROPORTION,
                    help='warmup proportion of training steps')
    ap.add_argument('--input_format', choices=READERS.keys(),
                    default=list(READERS.keys())[0],
                    help='input file format')
    ap.add_argument('--multiclass', default=False, action='store_true',
                    help='task has exactly one label per text')
    ap.add_argument('--output_file', default=None, metavar='FILE',
                    help='save model to file')
    ap.add_argument('--save_predictions', default=False, action='store_true',
                    help='save predictions and labels for dev set, or for test set if provided')
    ap.add_argument('--load_model', default=None, metavar='FILE',
                    help='load model from file')
    ap.add_argument('--log_file', default="train.log", metavar='FILE',
                    help='log parameters and performance to file')
    ap.add_argument('--test_log_file', default="test.log", metavar='FILE',
                    help='log parameters and performance on test set to file')
    return ap



def load_pretrained(options):
    name = options.model_name
    config = AutoConfig.from_pretrained(name)
    config.return_dict = True
    tokenizer = AutoTokenizer.from_pretrained(name, config=config)
    model = TFAutoModel.from_pretrained(name, config=config)
    #model = TFAutoModelForSequenceClassification.from_pretrained(name, config=config)

    if options.seq_len > config.max_position_embeddings:
        warning(f'--seq_len ({options.seq_len}) > max_position_embeddings '
                f'({config.max_position_embeddings}), using latter')
        options.seq_len = config.max_position_embeddings

    return model, tokenizer, config


def get_optimizer(num_train_examples, options):
    steps_per_epoch = math.ceil(num_train_examples / options.batch_size)
    num_train_steps = steps_per_epoch * options.epochs
    num_warmup_steps = math.floor(num_train_steps * options.warmup_proportion)

    # Mostly defaults from transformers.optimization_tf
    optimizer, lr_scheduler = create_optimizer(
        options.lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        min_lr_ratio=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay_rate=0.01,
        power=1.0,
    )
    return optimizer


def build_classifier(pretrained_model, num_labels, optimizer, options, MAX_LINES=500):
    seq_len = options.seq_len
    # Document-level input
    doc_input = Input(shape=(MAX_LINES,))

    ## Transformer block
    # Transformer inputs
    input_ids = Input(
        shape=(seq_len,), dtype='int32', name='input_ids')
    attention_mask = Input(
        shape=(seq_len,), dtype='int32', name='attention_mask')
    inputs = [input_ids, attention_mask]

    pretrained_outputs = pretrained_model(inputs)
    pooled_output = pretrained_outputs['last_hidden_state'][:,0,:] #CLS
    #import pdb; pdb.set_trace()
    pooled_output_ = Reshape((-1,768))(pooled_output)
    ## End of transformer block

    encoded_lines = TimeDistributed(pooled_output_)(doc_input)
    # encoded_lines shape (hopefully): [n_docs, n_lines, emb_dim]
    lstm = LSTM(768)(encoded_lines, return_sequences=True)
    output = TimeDistributed(Dense(2, activation='softmax'))(lstm)
    loss = CategoricalCrossentropy()
    F1Score(name='f1', num_classes=num_labels, average='micro')
    """
    # TODO consider Dropout here
    if options.multiclass:
        output = Dense(num_labels, activation='softmax')(pooled_output)
        loss = CategoricalCrossentropy()
        metrics = [CategoricalAccuracy(name='acc')]
    else:
        denselayer = Dense(seq_len, activation='softmax') #Anna muokkasi tästä
        output = TimeDistributed(denselayer)(inputs)
        loss = BinaryCrossentropy()
        metrics = [
            #F1Score(name='f1_th0.3', num_classes=num_labels, average='micro', threshold=0.3),
            F1Score(name='f1_th0.4', num_classes=num_labels, average='micro', threshold=0.4)#,
            #F1Score(name='f1_th0.5', num_classes=num_labels, average='micro', threshold=0.5),
            #AUC(name='auc', multi_label=True)
        ]
    """
    #output = pretrained_outputs # test
    model = Model(
        inputs=inputs,
        outputs=[output]
    )
#    model.summary()
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model


@timed
def load_data(fn, options, max_chars=None):
    read = get_reader(options.input_format)
    texts, labels, lengths = [], [], []
    with open(fn) as f:
        for ln, (text, text_labels) in enumerate(read(f, fn), start=1):
            if options.multiclass and not text_labels:
                raise ValueError(f'missing label on line {ln} in {fn}: {l}')
            elif options.multiclass and len(text_labels) > 1:
                raise ValueError(f'multiple labels on line {ln} in {fn}: {l}')
#            texts.append(text[:max_chars])
            texts.append(text[:512])
#            print(text)
            labels.append(text_labels[:512])
            lengths.append(len(text))
#    print(texts)
#    texts = pad_sequences(texts)
#    labels = pad_sequences(labels)
#    print(texts[0])
    print(f'loaded {len(texts)} examples from {fn}', file=sys.stderr)
    return texts, labels


class DataGenerator(Sequence):
    def __init__(self, data_path, tokenize_func, options, max_chars=None, label_encoder=None):
        texts, labels = load_data(data_path, options, max_chars=max_chars)
        self.num_examples = len(texts)
        self.batch_size = options.batch_size
        #self.seq_len = options.seq_len
        self.X = tokenize_func(texts)

        if label_encoder is None:
            self.label_encoder = MultiLabelBinarizer()
            self.label_encoder.fit(labels)
        else:
            self.label_encoder = label_encoder

        self.Y = self.label_encoder.transform(labels)
        self.num_labels = len(self.label_encoder.classes_)

        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(self.num_examples)
        np.random.shuffle(self.indexes)

    def __len__(self):
        return self.num_examples//self.batch_size

    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_X = {}
        for key in self.X:
            batch_X[key] = np.empty((self.batch_size, *self.X[key].shape[1:]))
            for j, idx in enumerate(batch_indexes):
                batch_X[key][j] = self.X[key][idx]

        batch_y = np.empty((self.batch_size, *self.Y.shape[1:]), dtype=int)
        for j, idx in enumerate(batch_indexes):
            batch_y[j] = self.Y[idx]

        return batch_X, batch_y


def make_tokenization_function(tokenizer, options):
    seq_len = options.seq_len
    @timed
    def tokenize(text, lines):
        lines = lines
        tokenized = tokenizer(
            text,
            max_length=seq_len,
            truncation=True,
            padding='max_length',
            return_tensors='np'
        )
#        print("padded length",len(pad_sequences(tokenized['input_ids'], maxlen=seq_len)))
        input_ids = pad_sequences(tokenized['input_ids'], maxlen=seq_len)
#        print('orkkis')
#        print(input_ids[0])
#        print(input_ids)
        attention_mask = pad_sequences(tokenized['attention_mask'], maxlen=seq_len)
#        print(attention_mask)
        input_ids = np.pad(input_ids, ((0,lines-len(input_ids)), (0,0)), 'constant', constant_values=1)
        attention_mask = np.pad(attention_mask, ((0,lines-len(attention_mask)),(0,0)))

#        print('uusi')
#        print(input_ids)
#        print(attention_mask)
#        print("padded length",len(pad_sequences(input_ids, maxlen=seq_len)))
#        print("shape", np.shape(input_ids))
        # Return dict b/c Keras (2.3.0-tf) DataAdapter doesn't apply
        # dict mapping to transformer.BatchEncoding inputs
#        return input_ids, attention_mask
        return {
#            'input_ids': pad_sequences(tokenized['input_ids'], maxlen=seq_len),
            'input_ids': pad_sequences(input_ids, maxlen=seq_len),
#            'token_type_ids': np.pad(tokenized['token_type_ids'], ((0,seq_len-len(attention_mask)),(0,0))),
#            'attention_mask': pad_sequences(tokenized['attention_mask'], maxlen=seq_len),
            'attention_mask': pad_sequences(attention_mask, maxlen=seq_len)
        }# palauttaa dictin, jossa 2 avainta, joilla arvoina listat
#        return pad_sequences(dict, maxlen=seq_len)
    return tokenize


@timed
def prepare_classifier(num_train_examples, num_labels, options):
    pretrained_model, tokenizer, config = load_pretrained(options)
    optimizer = get_optimizer(num_train_examples, options)
    model = build_classifier(pretrained_model, num_labels, optimizer, options)
    return model, tokenizer, optimizer


def optimize_threshold(model, train_X, train_Y, test_X, test_Y, options=None, epoch=None, save_pred_to=None):
    labels_prob = model.predict(train_X, verbose=1)#, batch_size=options.batch_size)

    best_f1 = 0.
    print("Optimizing threshold...\nThres.\tPrec.\tRecall\tF1")
    for threshold in np.arange(0.1, 0.9, 0.05):
        labels_pred = lil_matrix(labels_prob.shape, dtype='b')
        labels_pred[labels_prob>=threshold] = 1
        precision, recall, f1, _ = precision_recall_fscore_support(train_Y, labels_pred, average="micro")
        print("%.2f\t%.4f\t%.4f\t%.4f" % (threshold, precision, recall, f1), end="")
        if f1 > best_f1:
            print("\t*")
            best_f1 = f1
            #best_f1_epoch = epoch
            best_f1_threshold = threshold
        else:
            print()

    #print("Current F_max:", best_f1, "epoch", best_f1_epoch+1, "threshold", best_f1_threshold, '\n')
    #print("Current F_max:", best_f1, "threshold", best_f1_threshold, '\n')

    test_labels_prob = model.predict(test_X, verbose=1)#, batch_size=options.batch_size)
    test_labels_pred = lil_matrix(test_labels_prob.shape, dtype='b')
    test_labels_pred[test_labels_prob>=best_f1_threshold] = 1
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_Y, test_labels_pred, average="micro")
    if epoch:
        epoch_str = ", epoch %d" % epoch
    else:
        epoch_str = ""
    print("\nValidation/Test performance at threshold %.2f%s: Prec. %.4f, Recall %.4f, F1 %.4f" % (best_f1_threshold, epoch_str, test_precision, test_recall, test_f1))

    if save_pred_to is not None:
        if epoch is None:
            print("Saving predictions to", save_pred_to+".*.npy" % epoch)
            np.save(save_pred_to+".preds.npy", test_labels_pred.toarray())
            np.save(save_pred_to+".gold.npy", test_Y)
            #np.save(save_pred_to+".class_labels.npy", label_encoder.classes_)
        else:
            print("Saving predictions to", save_pred_to+"-epoch%d.*.npy" % epoch)
            np.save(save_pred_to+"-epoch%d.preds.npy" % epoch, test_labels_pred.toarray())
            np.save(save_pred_to+"-epoch%d.gold.npy" % epoch, test_Y)
            #np.save(save_pred_to+"-epoch%d.class_labels.npy" % epoch, label_encoder.classes_)

    return test_f1, best_f1_threshold, test_labels_pred


def test_threshold(model, test_X, test_Y, threshold=0.4, options=None, epoch=None):
    test_labels_prob = model.predict(test_X, verbose=1)#, batch_size=options.batch_size)
    test_labels_pred = lil_matrix(test_labels_prob.shape, dtype='b')
    test_labels_pred[test_labels_prob>=threshold] = 1
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_Y, test_labels_pred, average="micro")
    if epoch:
        epoch_str = ", epoch %d" % epoch
    else:
        epoch_str = ""
    print("\nValidation/Test performance at threshold %.2f%s: Prec. %.4f, Recall %.4f, F1 %.4f" % (threshold, epoch_str, test_precision, test_recall, test_f1))
    return test_f1, threshold, test_labels_pred


def test_auc(model, test_X, test_Y):
    labels_prob = model.predict(test_X, verbose=1)
    return roc_auc_score(test_Y, labels_prob, average = 'micro')


class Logger:
    def __init__(self, filename, model, params):
        self.filename = filename
        self.model = model
        self.log = dict([('p%s'%p, v) for p, v in params.items()])

    def record(self, epoch, logs):
        for k in logs:
            self.log['_%s' % k] = logs[k]
        self.log['_Epoch'] = epoch
        self.write()

    def write(self):
        file_exists = isfile(self.filename)
        with open(self.filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, sorted(self.log.keys()))
            if not file_exists:
                print("Creating log file", self.filename, flush=True)
                writer.writeheader()
            writer.writerow(self.log)


class EvalCallback(Callback):
    def __init__(self, model, train_X, train_Y, dev_X, dev_Y, test_X=None, test_Y=None, logfile="train.log", test_logfile=None, save_pred_to=None, params={}):
        self.model = model
        self.train_X = train_X
        self.train_Y = train_Y
        self.dev_X = dev_X
        self.dev_Y = dev_Y
        self.test_X = test_X
        self.test_Y = test_Y
        self.logger = Logger(logfile, self.model, params)
        if test_logfile is not None:
            print("Setting up test set logging to", test_logfile, flush=True)
            self.test_logger = Logger(test_logfile, self.model, params)
        if save_pred_to is not None:
            self.save_pred_to = save_pred_to
        else:
            self.save_pred_to = None

    def on_epoch_end(self, epoch, logs={}):
        print("Validation set performance:")
        logs['f1'], _, _ = optimize_threshold(self.model, self.train_X, self.train_Y, self.dev_X, self.dev_Y, epoch=epoch)
        logs['rocauc'] = test_auc(self.model, self.dev_X, self.dev_Y)
        print("AUC dev:", logs['rocauc'])

        self.logger.record(epoch, logs)
        if self.test_X is not None:
            print("Test set performance:")
            test_f1, _, _ = optimize_threshold(self.model, self.train_X, self.train_Y, self.test_X, self.test_Y, epoch=epoch, save_pred_to=self.save_pred_to)
            auc = test_auc(self.model, self.test_X, self.test_Y)
            print("AUC test:", auc)
            self.test_logger.record(epoch, {'f1': test_f1, 'rocauc': auc})


def main(argv):
    init_tf_memory()
    options = argparser().parse_args(argv[1:])

    train_texts, train_labels = load_data(options.train, options, max_chars=25000) # train_texts: list of docs, which are list of lines; train_labels same format
    dev_texts, dev_labels = load_data(options.dev, options, max_chars=25000)
    if options.test is not None:
        test_texts, test_labels = load_data(options.test, options, max_chars=25000)
    num_train_examples = len(train_texts)
    num_lines = max([len(i) for i in train_texts])
    label_encoder = MultiLabelBinarizer() # TODO: don't use multilabel
    label_encoder.fit(train_labels) #
    train_Y = label_encoder.transform(train_labels)
    train_Y = pad_sequences(train_Y, maxlen=512) # TODO: replace with cropping, apply to_categorical
    print("train_Y shape", np.shape(train_Y))
    dev_Y = label_encoder.transform(dev_labels)
    dev_Y = pad_sequences(dev_Y, maxlen=512)
    if options.test is not None:
        test_Y = label_encoder.transform(test_labels)
        test_Y = pad_sequences(test_Y, maxlen=512)
    num_labels = len(label_encoder.classes_)
 #   print(num_labels)
    classifier, tokenizer, optimizer = prepare_classifier(
        num_train_examples,
        num_labels,
        options
    )

    tokenize = make_tokenization_function(tokenizer, options)
#    train_X = tokenize(train_texts, 512)
#    print(np.shape(train_X))
#    dev_X = tokenize(dev_texts)
    train_X = {'input_ids': [], 'attention_mask': [] # TODO: mode input_ids/attention_mask dict inside data structure
              # ,'token_type_ids': []
    }
    dev_X = {'input_ids': [], 'attention_mask': []
             #, 'token_type_ids': []
    }
    input_ids = []
    attention_mask = []
    for train_doc in train_texts:
        input_ids.append([])
        attention_mask.append([])
        MAX_LINES = 500
        for i, line in enumerate(train_doc):
            if i >= MAX_LINES:
                break
            input = tokenize(line, options.seq_len)
            input_ids[-1].append(input['token_ids'])
            attention_mask[-1].append(input['attention_mask'])

        # Pad lines
        for i in range(MAX_LINES-len(input_ids)):
            input_ids[-1].append(input['token_ids']*0)
            attention_mask[-1].append(input['attention_mask']*0)

    train_X = {'input_ids': np.array(input_ids), 'attention_mask': np.array(attention_mask)}
    #    train_X['input_ids'].append(
    #['input_ids'])
    #  train_X['token_type_ids'].append(tokenize(train, num_lines)['token_type_ids'])
    #    train_X['attention_mask'].append(tokenize(train, num_lines)['attention_mask'])
    print('input_ids shape', np.shape(train_X['input_ids']))
    for dev in dev_texts:
        dev_X['input_ids'].append(tokenize(dev, num_lines)['input_ids'])
     #   dev_X['token_type_ids'].append(tokenize(dev, num_lines)['token_type_ids'])
        dev_X['attention_mask'].append(tokenize(dev, num_lines)['attention_mask'])
    train_X['input_ids'] = np.asarray(train_X['input_ids'])
    train_X['attention_mask'] = np.asarray(train_X['attention_mask'])
    dev_X['input_ids'] = np.asarray(dev_X['input_ids']).astype('float32')
    dev_X['attention_mask'] = np.asarray(dev_X['attention_mask']).astype('float32')

    #print("TRAIN SHAPE:", np.shape(train_X))
    #print(train_X)
    #train_gen = DataGenerator(options.train, tokenize, options, max_chars=25000)
    #dev_gen = DataGenerator(options.dev, tokenize, options, max_chars=25000, label_encoder=train_gen.label_encoder)

    if options.test is not None:
        test_X = {'input_ids':[],'attention_mask':[]}
        for test in test_texts:
            test_X['input_ids'].append(tokenize(test, num_lines)['input_ids'])
    #        test_X['token_type_ids'].append(tokenize(test, num_lines)['token_type_ids'])
            test_X['attention_mask'].append(tokenize(test, num_lines)['attention_mask'])

    if options.load_model is not None:
        classifier.load_weights(options.load_model)

        print("Evaluating on dev set...")
        f1, th, dev_pred = optimize_threshold(classifier, train_X, train_Y, dev_X, dev_Y, options)
        print("AUC dev:", test_auc(classifier, dev_X, dev_Y))

        if options.test is not None:
            print("Evaluating on test set...")
            test_f1, test_th, test_pred = optimize_threshold(classifier, train_X, train_Y, test_X, test_Y, options, save_pred_to=options.load_model)
            np.save(options.load_model+".class_labels.npy", label_encoder.classes_)
            #test_f1, test_th, test_pred = optimize_threshold(classifier, train_X, train_Y, test_X, test_Y, options)
            print("AUC test:", test_auc(classifier, test_X, test_Y))

        #f1, th, dev_pred = test_threshold(classifier, dev_X, dev_Y, threshold=0.4)
        return


    callbacks = [] #[ModelCheckpoint(options.output_file+'.{epoch:02d}', save_weights_only=True)]
    if options.test is not None and options.test_log_file is not None:
        print("Initializing evaluation with dev and test set...")
        callbacks.append(EvalCallback(classifier, train_X, train_Y, dev_X, dev_Y, test_X=test_X, test_Y=test_Y,
                                    logfile=options.log_file,
                                    test_logfile=options.test_log_file,
                                    save_pred_to=options.output_file,
                                    params={'LR': options.lr, 'N_epochs': options.epochs, 'BS': options.batch_size}))
    else:
        print("Initializing evaluation with dev set...")
        callbacks.append(EvalCallback(classifier, train_X, train_Y, dev_X, dev_Y,
                                    logfile=options.log_file,
                                    params={'LR': options.lr, 'N_epochs': options.epochs, 'BS': options.batch_size}))

    history = classifier.fit(
        train_X,
        train_Y,
        epochs=options.epochs,
        batch_size=options.batch_size,
        validation_data=(np.asarray(dev_X), dev_Y),
        callbacks=callbacks
    )
    """
    history = classifier.fit_generator(
        train_gen,
        steps_per_epoch=len(train_gen),
        validation_data=dev_gen,
        initial_epoch=0,
        epochs=options.epochs,
        callbacks=callbacks
    )"""


    try:
        if options.output_file:
            print("Saving model to %s" % options.output_file)
            classifier.save_weights(options.output_file)
    except:
        pass

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
