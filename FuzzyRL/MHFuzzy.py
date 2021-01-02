import numpy as np
import matplotlib.pyplot as plt

class MHMember:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.center = (b+c)/2

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
        outputs = set()
        for i in range(len(self.rule_list)):
            outputs.add(self.rule_list[i].output_list[0])
        centers = []
        weights = []
        for i in outputs:
            weights.append(max([self.rule_list[j].evaluate(inputs) for j in range(len(self.rule_list)) if self.rule_list[j].output_list[0].center==i.center]))
            centers.append(i.center)
        output = (sum([centers[i] * weights[i] for i in range(len(centers))]))/(sum(weights))
        return output

if __name__=='__main__':
    small = MHMember(0, 0, 0.1, 0.15)
    medium = MHMember(0.1, 0.2, 0.3, 0.4)
    big = MHMember(0.35, 0.5, 0.5, 0.5)
    close = MHMember(0, 0, 0, 10)
    far = MHMember(5, 10, 100, 100)
    rule1 = MHRule([close], [small])
    rule2 = MHRule([far], [big])
    fis = MHFIS([rule1, rule2])
    print(fis.output([20]))

