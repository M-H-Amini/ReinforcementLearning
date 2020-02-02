import numpy as np
import matplotlib.pyplot as plt

class MHMember:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __call__(self, x):
        if x < self.a or x > self.d:
            return 0
        elif x >= self.b and x <= self.c:
            return 1
        elif x > self.c:
            return (self.d - x)/(self.d - self.c)
        else:
            return (x - self.a)/(self.b - self.a)

class MHRule:
    def __init__(self, input_list, output_list):
        self.input_list = input_list
        self.output_list = output_list

    def evaluate(self, inputs):
        return min([self.input_list[i](inputs[i]) for i in range(len(self.input_list))])

class MHFIS:
    def __init__(self, rule_list):
        self.rule_list = rule_list

    def output(self, inputs):
        return max([self.rule_list[i].evaluate(inputs) for i in range(len(self.rule_list))])

if __name__=='__main__':
    small = MHMember(0, 0, 0, 0.15)
    medium = MHMember(0.1, 0.2, 0.3, 0.4)
    big = MHMember(0.35, 0.5, 0.5, 0.5)
    close = MHMember(0, 0, 0, 10)
    far = MHMember(8, 12, 100, 100)
    rule1 = MHRule([close], [small])
    rule2 = MHRule([far], [big])
    fis = MHFIS([rule1, rule2])
    print(fis.output([1]))

