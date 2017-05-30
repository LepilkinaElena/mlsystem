"""Implement the command line 'mlsystem' tool."""
import os
import sys
import tempfile
import json
from optparse import OptionParser, OptionGroup, Option, OptionValueError
from copy import copy
import contextlib
import mlsystem.util.multitool
import mlsystem.dataset.dataset as dataset
import mlsystem.model.model as model
import mlsystem.model.dnn_model as dnn_model
import mlsystem.model.dnn_wide as dnn_wide
import mlsystem.model.cnn_model as cnn_model
import mlsystem.model.rnn_model as rnn_model
import mlsystem.model.birnn_model as birnn_model
import mlsystem.model.perceptron as perceptron
from mlsystem.parser.parser import RuleParser
from tensorflow.contrib.learn.python.learn.datasets import base

def pair(arg):
    return [x for x in arg.split(',')]

def check_pair(value):
    try:
        return pair(value)
    except ValueError:
        raise OptionValueError ("invalid value %s" %  value)

def action_start_train(name, args):
    """start a new development server"""

    parser = OptionParser("""\
%s [options] <instance path>""" % name)
    parser.add_option("", "--runs", type=str, action="append",
                      dest="runs", default=[],
                      help="runs for getting data")
    parser.add_option("", "--rule-file", dest="rule_file", type=str,
                      help="file with description of rule for best result", 
                      default=None)

    parser.add_option("", "--option", dest="option", type=str,
                      help="option for training", 
                      default=None)
    parser.add_option("", "--values", dest="values", type=str,
                      help="values assotiated with runs", 
                      default=None)
    parser.add_option("", "--pass", dest="pass_name", type=str,
                      help="pass for option tuning", 
                      default=None)

    (opts, args) = parser.parse_args(args)
    runs = []
    for run in opts.runs:
        runs.append(check_pair(run))
        print(runs)

    # Parse file:
    if not opts.rule_file:
        parser.error("File with rule description is nessecary!")
    if not opts.option:
        parser.error("Option is nessecary!")
    if not opts.values and runs:
        parser.error("Values for option in runs are nessecary!")
    if not opts.pass_name:
        parser.error("Pass name is nessecary!")
    
    print(opts.values)
    values = check_pair(opts.values)
    with open(opts.rule_file) as rule_content:
        rule = rule_content.read().replace('\n', '').replace(' ', '')
        rule_parser = RuleParser()
        stack = rule_parser.parse(rule)
        dataset_producer = dataset.DatasetProducer(runs, values, rule_parser, stack, opts.pass_name)
        test_dataset = dataset_producer.next_batch(100, True)
        validation_dataset = dataset_producer.next_batch(1)
        train_dataset = dataset_producer
        datasets = base.Datasets(train = train_dataset, validation = validation_dataset, test = test_dataset)
        dnn_wide.model(datasets, len(values))
        
tool = mlsystem.util.multitool.MultiTool(locals())

def main(*args, **kwargs):
    return tool.main(*args, **kwargs)

if __name__ == '__main__':
    main()