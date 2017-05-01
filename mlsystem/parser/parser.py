from pyparsing import CaselessLiteral, Literal, Forward, nums, Combine, Optional,\
                      Word, alphas, oneOf, Group, ZeroOrMore

import math
import operator
import re

class RuleParser(object):
    """
    Rule parser.
    """
    def pushFirst(self, strg, loc, toks):
        self.exprStack.append(toks[0])

    def pushUMinus(self, strg, loc, toks):
        if toks and toks[0] == '-':
            self.exprStack.append('unary -')

    def __init__(self):
        exec_time_const = CaselessLiteral("exec_time")
        code_size_const = CaselessLiteral("code_size")
        llc_misses_const = CaselessLiteral("llc_misses")
        exec_time_p_const = CaselessLiteral("program_exec_time")
        code_size_p_const = CaselessLiteral("program_code_size")
        llc_misses_p_const = CaselessLiteral("program_llc_misses")

        self.constants = ["exec_time", "code_size", "llc_misses", 
                          "program_exec_time", "program_code_size", "program_llc_misses"]
        point = Literal(".")
        fnumber = Combine(Word("+-" + nums, nums) + Optional(point + Optional(Word(nums))))
        lpar = Literal("(").suppress()
        rpar = Literal(")").suppress()
        plus = Literal("+")
        minus = Literal("-")
        mul = Literal("*")
        div = Literal("/")
        
        expop = Literal("^")
        ident_const = exec_time_const | code_size_const | llc_misses_const | \
                exec_time_p_const | code_size_p_const | llc_misses_p_const

        ident = Word(alphas, alphas + nums + "_$")

        multop = mul | div
        addop = plus | minus
        expr = Forward()
        atom = ((Optional(oneOf("- +")) + 
                (ident + lpar +expr + rpar | ident_const | fnumber).setParseAction(self.pushFirst))
                | Optional(oneOf("- +")) + Group(lpar + expr + rpar)).setParseAction(self.pushUMinus)
        factor = Forward()
        factor << atom + ZeroOrMore((expop + factor).setParseAction(self.pushFirst))
        term = factor + ZeroOrMore((multop + factor).setParseAction(self.pushFirst))
        expr << term + ZeroOrMore((addop + term).setParseAction(self.pushFirst))
        self.bnf = expr

        self.opn = {"+": operator.add,
                    "-": operator.sub,
                    "*": operator.mul,
                    "/": operator.truediv,
                    "^": operator.pow}
        self.fn = {"sin": math.sin,
                   "cos": math.cos,
                   "tan": math.tan,
                   "exp": math.exp,
                   "abs": abs}

    def parse(self, rule_string):
        self.exprStack = []
        rule = re.match(r'(min|max)\s*\((.+)\)', rule_string)
        if rule:
            self.criteria = rule.group(1)
            print(rule.group(2))
            results = self.bnf.parseString(rule.group(2), True)
            return self.exprStack

    def eval(self, stack, value_dict):
        op = stack.pop()
        if op == "unary -":
            return -self.eval(stack, value_dict)
        if op in "+-*/^":
            op_sec = self.eval(stack, value_dict)
            op_first = self.eval(stack, value_dict)
            return self.opn[op](op_first, op_sec)
        elif op in self.constants:
            return value_dict[op]
        elif op in self.fn:
            return self.fn[op](self.eval(stack, value_dict))
        else:
            return float(op)

#val = self.eval(self.exprStack[:])