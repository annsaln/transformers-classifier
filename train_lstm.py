#!/usr/bin/env python3

import sys
import math
import numpy as np
from os.path import isfile
import csv
import json

from scipy.sparse import lil_matrix

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dropout, Dense, Layer, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, AUC
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
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
DEFAULT_MAX_LINES = 100

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
    ap.add_argument('--max_lines', metavar='INT', type=int,
                    default=DEFAULT_MAX_LINES)
    ap.add_argument('--warmup_proportion', metavar='FLOAT', type=float,
                    default=DEFAULT_WARMUP_PROPORTION,
                    help='warmup proportion of training steps')
    ap.add_argument('--input_format', choices=READERS.keys(),
                    default=list(READERS.keys())[0],
                    help='input file format')
    ap.add_argument('--multiclass', default=False, action='store_true',
                    help='task has exactly one label per text')
    ap.add_argument('--multilabel', default=False, action='store_true',
                    help='task has more than two possible labels')
    ap.add_argument('--output_file', default=None, metavar='FILE',
                    help='save model to file')
    ap.add_argument('--save_predictions', default=False, action='store_true',
                    help='save predictions and labels for dev set, or for test set if provided')
    ap.add_argument('--load_model', default=None, metavar='FILE',
                    help='load model from file')
    ap.add_argument('--log_file', default="train.log", metavar='FILE',
                    help='log parameters and performance to file')
    ap.add_argument('--threshold', metavar='FLOAT', type=float, default=None, 
                    help='fixed threshold for multilabel prediction')
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


def build_classifier(pretrained_model, num_labels, optimizer, options, num_train_examples, train_doc_ids):
    seq_len = options.seq_len
    MAX_LINES = options.max_lines
    train_doc_ids = train_doc_ids
#    loaded_model = options.load_model
    input_ids = Input(
        shape=(seq_len,), dtype='int32', name='input_ids')
#    token_type_ids = Input(
#        shape=(seq_len,), dtype='int32', name='token_type_ids')
    attention_mask = Input(
        shape=(seq_len,), dtype='int32', name='attention_mask')
#    inputs = [input_ids, attention_mask, token_type_ids]
    inputs = [input_ids, attention_mask]

    pretrained_outputs = pretrained_model(inputs)
    #pooled_output = pretrained_outputs[1]        
#    pooled_output = pretrained_outputs['last_hidden_state'][:,0,:] #CLS
    transformer_output = pretrained_outputs['last_hidden_state'][:,0,:] #CLS
#    transformer_output = Dense(num_labels, activation='softmax')(pooled_output)
    print("TRANSFORMER OUTPUT SHAPE", np.shape(transformer_output))

    '''
    # TODO consider Dropout here
    if options.multiclass:
        output = Dense(num_labels, activation='softmax')(pooled_output)
        loss = CategoricalCrossentropy()
        metrics = [CategoricalAccuracy(name='acc')]
    else:
        output = Dense(num_labels, activation='sigmoid')(pooled_output)
        loss = BinaryCrossentropy()
        metrics = [
            #F1Score(name='f1_th0.3', num_classes=num_labels, average='micro', threshold=0.3),
            #F1Score(name='f1_th0.4', num_classes=num_labels, average='micro', threshold=0.4)#,
            F1Score(name='f1_th0.5', num_classes=num_labels, average='micro', threshold=0.5)#,
            #AUC(name='auc', multi_label=True)
        ]
    #output = pretrained_outputs # test
    '''
    last_doc_ID = 0
    lstm_input_data = np.zeros((num_train_examples, MAX_LINES, 768))
    print("LSTM INPUT SHAPE", np.shape(lstm_input_data))
    current_doc_idx = 0
    current_line_idx = 0
    for line_idx in range(8280): # fixed dimension, to be updated
        if train_doc_ids[line_idx] == last_doc_ID:
            if current_line_idx >= MAX_LINES:
                break
            lstm_input_data[current_doc_idx,:current_line_idx] = transformer_output[line_idx]
            current_line_idx += 1
            # check that current_line_idx < MAX_LINES
            # insert transformer_output[line_idx] into lstm_input_data[current_doc_idx, current_line_idx]
        else:
            current_line_idx = 0
            current_doc_idx += 1
            last_doc_ID = doc_IDs[line_idx]
            lstm_input_data[current_doc_idx,:current_line_idx] = transformer_output[line_idx]
#            np.append(lstm_input_data[current_doc_idx, current_line_idx],inputs[line_idx])
            # then insert as above
    
    lstm = LSTM(768)(lstm_input_data)
    output = Dense(2, activation='softmax')(lstm)
    loss = BinaryCrossentropy()
    metrics = [
        F1Score(name='f1_th0.5', num_classes=num_labels, average='micro', threshold=0.5)
        ]
    
    model = Model(
        inputs=inputs,
        outputs=output
    )
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model


@timed
def load_data(fn, options, max_chars=None):
    read = get_reader(options.input_format)
    texts, labels = [], []
    with open(fn) as f:
        for ln, (text, text_labels) in enumerate(read(f, fn), start=1):
            if options.multiclass and not text_labels:
                raise ValueError(f'missing label on line {ln} in {fn}: {l}')
            elif options.multiclass and len(text_labels) > 1:
                raise ValueError(f'multiple labels on line {ln} in {fn}: {l}')
            if len(text) != len(text_labels):
                continue
                #raise ValueError(f'labels and lines do not match')
            texts.append(text)
            labels.append(text_labels)
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
    def tokenize(text):
        tokenized = tokenizer(
            text,
            max_length=seq_len,
            truncation=True,
            padding=True,
            return_tensors='np'
        )
        # Return dict b/c Keras (2.3.0-tf) DataAdapter doesn't apply
        # dict mapping to transformer.BatchEncoding inputs
        return {
            'input_ids': tokenized['input_ids'],
#            'token_type_ids': tokenized['token_type_ids'],
            'attention_mask': tokenized['attention_mask'],
        }
    return tokenize


@timed
def prepare_classifier(num_train_examples, num_labels, options, train_doc_ids):
    optimizer = get_optimizer(num_train_examples, options)
    pretrained_model, tokenizer, config = load_pretrained(options)
    model = build_classifier(pretrained_model, num_labels, optimizer, options, num_train_examples, train_doc_ids)
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


def test_threshold(model, test_X, test_Y, threshold=0.4, options=None, epoch=None, return_auc=False, save_pred_to=None):
    test_labels_prob = model.predict(test_X, verbose=1)#, batch_size=options.batch_size)
    test_labels_pred = lil_matrix(test_labels_prob.shape, dtype='b')
    test_labels_pred[test_labels_prob>=threshold] = 1
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_Y, test_labels_pred, average="micro")
    if epoch:
        epoch_str = ", epoch %d" % epoch
    else:
        epoch_str = ""
    print("\nValidation/Test performance at threshold %.2f%s: Prec. %.4f, Recall %.4f, F1 %.4f" % (threshold, epoch_str, test_precision, test_recall, test_f1))
    if save_pred_to is not None:
        if epoch is None:
            print("Saving predictions to", save_pred_to+".*.npy")
            np.save(save_pred_to+".preds.npy", test_labels_pred.toarray())
            np.save(save_pred_to+".gold.npy", test_Y)
            #np.save(save_pred_to+".class_labels.npy", label_encoder.classes_)
        else:
            print("Saving predictions to", save_pred_to+"-epoch%d.*.npy" % epoch)
            np.save(save_pred_to+"-epoch%d.preds.npy" % epoch, test_labels_pred.toarray())
            np.save(save_pred_to+"-epoch%d.gold.npy" % epoch, test_Y)
            #np.save(save_pred_to+"-epoch%d.class_labels.npy" % epoch, label_encoder.classes_)

    if return_auc:
        auc = roc_auc_score(test_Y, test_labels_prob, average = 'micro')
        return test_f1, threshold, test_labels_pred, auc
    else:
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
    MAX_LINES = options.max_lines
    train_texts, train_labels = load_data(options.train, options, max_chars=25000)
    print("TEXTS", len(train_texts), len(train_labels), "LINES")
    dev_texts, dev_labels = load_data(options.dev, options, max_chars=25000)
    if options.test is not None:
        test_texts, test_labels = load_data(options.test, options, max_chars=25000)
#    num_train_examples = len(train_texts)


    '''
    if options.multilabel:
        label_encoder = MultiLabelBinarizer()
    else:
        label_encoder = LabelEncoder()

    label_encoder.fit(train_labels)
    train_Y = label_encoder.transform(train_labels)
    train_Y = to_categorical(train_Y)
    dev_Y = label_encoder.transform(dev_labels)
    dev_Y = to_categorical(dev_Y)
    if options.test is not None:
        test_Y = label_encoder.transform(test_labels)
        test_Y = to_categorical(test_Y)

    num_labels = len(label_encoder.classes_)
    '''
    num_labels = 2
    labels = {'0':0, '1':1}
    train_Y = []
    lstm_train_Y = []
    for text in train_labels:
        lines = []
        for label in text:
            train_Y.append(labels[label])
            lines.append(labels[label])
        lstm_train_Y.append(lines)
    #print("y before padding",np.shape(train_Y))
    lstm_train_Y = np.array(lstm_train_Y)
    lstm_train_Y = to_categorical(pad_sequences(lstm_train_Y, maxlen=MAX_LINES, padding='post'), num_classes=2, dtype='int') # TODO: replace with cropping, apply to_categorical
    label_encoder = LabelEncoder()
    label_encoder.fit(train_Y)
    print("LSTM Y SHAPE", np.shape(lstm_train_Y))
#    train_Y = label_encoder.transform(train_labels)
    print("Y SHAPE:", np.shape(train_Y))
    train_Y = label_encoder.transform(train_Y)
    train_Y = to_categorical(train_Y)
#    print("train_Y shape", np.shape(train_Y))
#    dev_Y = label_encoder.transform(dev_labels)
    dev_Y = []
    lstm_dev_Y = []
    for text in dev_labels:
        lines = []
        for label in text:
            dev_Y.append(labels[label])
            lines.append(labels[label])
        lstm_dev_Y.append(lines)
    lstm_dev_Y = np.array(lstm_dev_Y)
    lstm_dev_Y = to_categorical(pad_sequences(lstm_dev_Y, maxlen=MAX_LINES, padding='post'),num_classes=2, dtype='int')
    dev_Y = label_encoder.transform(dev_Y)
    dev_Y = to_categorical(dev_Y)
    if options.test is not None:
#        test_Y = label_encoder.transform(test_labels)
        test_Y = []
        lstm_test_Y = []
        for text in test_labels:
            lines = []
            for label in text:
                test_Y.append(labels[label])
                lines.append(labels[label])
            lstm_test_Y.append(lines)
        lstm_test_Y = np.array(lstm_test_Y)
        lstm_test_Y = to_categorical(pad_sequences(lstm_test_Y, maxlen=MAX_LINES, padding='post'),num_classes=2, dtype='int')
        test_Y = label_encoder.transform(test_Y)
        test_Y = to_categorical(test_Y)
    num_train_examples = len(lstm_train_Y)
#    classifier, tokenizer, optimizer = prepare_classifier(
#        num_train_examples,
#        num_labels,
#        options
#    )

    train_X = []
    train_doc_ids = []
    for i, doc in enumerate(train_texts):
        for line in doc:
            train_X.append(line)
            train_doc_ids.append(i)
    print("TRAIN SHAPE", np.shape(train_X))
    dev_X = []
    dev_doc_ids = []
    for i, doc in enumerate(dev_texts):
        for line in doc:
            dev_X.append(line)
            dev_doc_ids.append(i)
    if options.test is not None:
            test_X = []
            test_doc_ids = []
            for i, doc in enumerate(test_texts):
                for line in doc:
                    test_X.append(line)
                    test_doc_ids.append(i)
    
    classifier, tokenizer, optimizer = prepare_classifier(
        num_train_examples,
        num_labels,
        options,
        train_doc_ids
    )
    
    classifier.load_weights(options.load_model)

    tokenize = make_tokenization_function(tokenizer, options)
    
    train_X = tokenize(train_X)
    dev_X = tokenize(dev_X)
    #train_gen = DataGenerator(options.train, tokenize, options, max_chars=25000)
    #dev_gen = DataGenerator(options.dev, tokenize, options, max_chars=25000, label_encoder=train_gen.label_encoder)
    if options.test is not None:
        test_X = tokenize(test_X)
    ''''
    last_doc_ID = 0
    lstm_input_data = np.zeros((num_train_examples, MAX_LINES, 768))
    current_doc_idx = 0
    for line_idx in range(transformer_output.shape[0]):
        if train_doc_ids[line_idx] == last_doc_ID:
            if current_line_idx >= MAX_LINES:
                break
            lstm_input_data[current_doc_idx, current_line_idx].append(transformer_output[line_idx])
            current_line_idx += 1
            # check that current_line_idx < MAX_LINES
            # insert transformer_output[line_idx] into lstm_input_data[current_doc_idx, current_line_idx]
        else:
            current_line_idx = 0
            current_doc_idx += 1
            last_doc_ID = doc_IDs[line_idx]
            lstm_input_data[current_doc_idx, current_line_idx].append(transformer_output[line_idx])
            # then insert as above
    '''
    '''
    input_ids = []
    attention_mask = []
    dev_doc_ids = []
    
    for train_doc in train_texts:
        input_ids.append([])
        attention_mask.append([])
        for i, line in enumerate(train_doc):
            if i >= MAX_LINES:
                break
            input = tokenize(line)
            input_ids[-1].extend(input['input_ids'])
            attention_mask[-1].extend(input['attention_mask'])

        # Pad lines
        for i in range(MAX_LINES-len(input_ids[-1])):
            input_ids[-1].extend(input['input_ids']*0)
            attention_mask[-1].extend(input['attention_mask']*0)
    print(np.shape(input_ids))
    train_X = {'input_ids': np.array(input_ids), 'attention_mask': np.array(attention_mask)}
    #    train_X['input_ids'].append(
    #['input_ids'])
    #  train_X['token_type_ids'].append(tokenize(train, num_lines)['token_type_ids'])
    #    train_X['attention_mask'].append(tokenize(train, num_lines)['attention_mask'])
    print('input_ids shape', np.shape(train_X['input_ids']))
    print('input_ids[0] shape', np.shape(train_X['input_ids'][0]))
    print(train_X['input_ids'][0])
#    print(train_X[:5])
    input_ids = []
    attention_mask = []
    for dev in dev_texts:
#        dev_X['input_ids'].append(tokenize(dev, num_lines)['input_ids'])
     #   dev_X['token_type_ids'].append(tokenize(dev, num_lines)['token_type_ids'])
#        dev_X['attention_mask'].append(tokenize(dev, num_lines)['attention_mask'])
        input_ids.append([])
        attention_mask.append([])
        MAX_LINES = 500
        for i, line in enumerate(dev):
            if i >= MAX_LINES:
                break
            input = tokenize(line)
            input_ids[-1].append(input['input_ids'])
            attention_mask[-1].append(input['attention_mask'])

        # Pad lines
        for i in range(MAX_LINES-len(input_ids[-1])):
            input_ids[-1].append(input['input_ids']*0)
            attention_mask[-1].append(input['attention_mask']*0)
    dev_X = {'input_ids': np.array(input_ids,dtype=np.float), 'attention_mask': np.array(attention_mask,dtype=np.float)}

#    print("TRAIN SHAPE:", np.shape(train_X))
    #print(train_X)
    #train_gen = DataGenerator(options.train, tokenize, options, max_chars=25000)
    #dev_gen = DataGenerator(options.dev, tokenize, options, max_chars=25000, label_encoder=train_gen.label_encoder)

    if options.test is not None:
        input_ids = []
        attention_mask = []
        for test in test_texts:
#        dev_X['input_ids'].append(tokenize(dev, num_lines)['input_ids'])
     #   dev_X['token_type_ids'].append(tokenize(dev, num_lines)['token_type_ids'])
#        dev_X['attention_mask'].append(tokenize(dev, num_lines)['attention_mask'])
            input_ids.append([])
            attention_mask.append([])
            MAX_LINES = 500
            for i, line in enumerate(test):
                if i >= MAX_LINES:
                    break
                input = tokenize(line)
                input_ids[-1].append(input['input_ids'])
                attention_mask[-1].append(input['attention_mask'])

            # Pad lines
            for i in range(MAX_LINES-len(input_ids[-1])):
                input_ids[-1].append(input['input_ids']*0)
                attention_mask[-1].append(input['attention_mask']*0)
        test_X = {'input_ids': np.array(input_ids,dtype=np.float), 'attention_mask': np.array(attention_mask,dtype=np.float)}
    '''
    
    '''
    if options.load_model is not None:
        classifier.load_weights(options.load_model)

        print("Evaluating on dev set...")
        if options.threshold is None:
            f1, th, dev_pred = optimize_threshold(classifier, train_X, train_Y, dev_X, dev_Y, options)
        else:
            f1, th, dev_pred = test_threshold(classifier, dev_X, dev_Y, threshold=options.threshold)
        print("AUC dev:", test_auc(classifier, dev_X, dev_Y))

        if options.test is not None:
            print("Evaluating on test set...")
            if options.threshold is None:
                test_f1, test_th, test_pred = optimize_threshold(classifier, train_X, train_Y, test_X, test_Y, options, save_pred_to=options.load_model)
            else:
                test_f1, test_th, test_pred = test_threshold(classifier, test_X, test_Y, threshold=options.threshold, save_pred_to=options.save_predictions)
            np.save(options.load_model+".class_labels.npy", label_encoder.classes_)
            #test_f1, test_th, test_pred = optimize_threshold(classifier, train_X, train_Y, test_X, test_Y, options)
            print("AUC test:", test_auc(classifier, test_X, test_Y))

        #f1, th, dev_pred = test_threshold(classifier, dev_X, dev_Y, threshold=0.4)
        return
    '''
    callbacks = [] #[ModelCheckpoint(options.output_file+'.{epoch:02d}', save_weights_only=True)]
    if options.threshold is None:
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
    else:
        print("Initializing early stopping criterion...")
        callbacks.append(EarlyStopping(monitor="val_f1_th0.5", verbose=1, patience=5, mode="max", restore_best_weights=True))


    history = classifier.fit(
        train_X,
        train_Y,
        epochs=options.epochs,
        batch_size=options.batch_size,
        validation_data=(dev_X, dev_Y),
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
    
    print("THIS IS AFTER TRANSFORMERS")
    if options.threshold is not None:
        if options.dev is not None:
            f1, _, _, rocauc = test_threshold(classifier, dev_X, dev_Y, return_auc=True, threshold=options.threshold)
            print("Restored best checkpoint, F1: %.6f, AUC: %.6f" % (f1, rocauc))

            logger = Logger(options.log_file, None, {'LR': options.lr, 'N_epochs': options.epochs, 'BS': options.batch_size})
            try:
                epoch = len(history.history['loss'])-1
            except:
                epoch = -1
            logger.record(epoch, {'f1': f1, 'rocauc': rocauc})

        if options.test is not None:
            '''
            for i, (test_X, test_Y) in enumerate(zip(test_Xs, test_Ys)):
                test_log_file = options.test_log_file.split(';')[i]
                print("Evaluating on test set %d..." % i)
                test_f1, test_th, test_pred, test_rocauc = test_threshold(classifier, test_X, test_Y, threshold=options.threshold, return_auc=True)
                print("AUC test:", test_rocauc)
                print("Logging to", test_log_file)
                test_logger = Logger(options.test_log_file.split(';')[i], None, {'LR': options.lr, 'N_epochs': options.epochs, 'BS': options.batch_size})
                try:
                    epoch = len(history.history['loss'])-1
                except:
                    epoch = -1
                test_logger.record(epoch, {'f1': test_f1, 'rocauc': test_rocauc})
            '''
            print("Evaluating on test set...")
            test_f1, test_th, test_pred = test_threshold(classifier, test_X, test_Y, threshold=options.threshold, return_auc=True)
            print("AUC test:", test_rocauc)
            print("Logging to", test_log_file)
            test_logger = Logger(options.test_log_file.split(';')[i], None, {'LR': options.lr, 'N_epochs': options.epochs, 'BS': options.batch_size})
            try:
                epoch = len(history.history['loss'])-1
            except:
                epoch = -1
            test_logger.record(epoch, {'f1': test_f1, 'rocauc': test_rocauc})

#    transformer_output = classifier.layers[-1]
#    print(np.shape(transformer_output))
    ''''
    last_doc_ID = 0
    lstm_input_data = np.zeros((num_train_examples, MAX_LINES, 768))
    current_doc_idx = 0
    for line_idx in range(transformer_output.shape[0]):
        if train_doc_ids[line_idx] == last_doc_ID:
            if current_line_idx >= MAX_LINES:
                break
            lstm_input_data[current_doc_idx, current_line_idx].append(transformer_output[line_idx])
            current_line_idx += 1
            # check that current_line_idx < MAX_LINES
            # insert transformer_output[line_idx] into lstm_input_data[current_doc_idx, current_line_idx]
        else:
            current_line_idx = 0
            current_doc_idx += 1
            last_doc_ID = doc_IDs[line_idx]
            lstm_input_data[current_doc_idx, current_line_idx].append(transformer_output[line_idx])
            # then insert as above
    '''
    '''
    try:
        if options.output_file:
            print("Saving model to %s" % options.output_file)
            classifier.save_weights(options.output_file)
    except:
        pass
    '''
    return 0
    

if __name__ == '__main__':
    sys.exit(main(sys.argv))

