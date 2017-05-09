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
from mlsystem.parser.parser import RuleParser

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

    
    print(opts.values)
    values = check_pair(opts.values)
    with open(opts.rule_file) as rule_content:
        rule = rule_content.read().replace('\n', '').replace(' ', '')
        rule_parser = RuleParser()
        stack = rule_parser.parse(rule)
        dataset_producer = dataset.DatasetProducer(runs, values, rule_parser, stack)
        data_set = dataset_producer.next_batch(100)
        print(data_set.size())

tool = mlsystem.util.multitool.MultiTool(locals())

def main(*args, **kwargs):
    return tool.main(*args, **kwargs)

if __name__ == '__main__':
    main()