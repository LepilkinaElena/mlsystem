"""Implement the command line 'mlsystem' tool."""
import os
import sys
import tempfile
import json
from optparse import OptionParser, OptionGroup, Option, OptionValueError
from copy import copy
import contextlib
import mlsystem.util.multitool
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

    (opts, args) = parser.parse_args(args)
    runs = []
    for run in opts.runs:
        runs.append(check_pair(run))
        print(runs)

    # Parse file:
    if not opts.rule_file:
        parser.error("file with rule description is nessecary!")
    with open(opts.rule_file) as rule_content:
        rule = rule_content.read().replace('\n', '').replace(' ', '')
        rule_parser = RuleParser()
        stack = rule_parser.parse(rule)
        print(stack)

tool = mlsystem.util.multitool.MultiTool(locals())

def main(*args, **kwargs):
    return tool.main(*args, **kwargs)

if __name__ == '__main__':
    main()