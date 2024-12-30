import sys
from information_gain import *
sys.path.append('')


class C45Lite:
    def __init__(self, tree=None):
        self.mytree = tree

    def majorityCnt(self, classList):
        p = {}
        for label in classList:
            if label in p:
                p[label] += 1
            else:
                p[label] = 1
        ma = None
        for key in p.keys():
            if ma is None or p[key] > p[ma]:
                ma = key

        return ma

    def chooseBestFeature(self, D):
        e = Ent(D, -1)
        idx = 0
        eu = Gain_utils(e, D, 0)
        for i in range(1, len(D[0]) - 1):
            cmp = Gain_utils(e, D, i)
            if cmp > eu:
                idx = i
                eu = cmp

        return idx

    def splitDataSet(self, D, a, v):
        n = []
        for row in D:
            if row[a] == v:
                n.append(np.delete(row, a))

        return np.array(n)

    def create_tree(self, D, labels, featLabels):
        classList = [label[-1] for label in D]
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(D[0]) == 1 or len(labels) == 0:
            return self.majorityCnt(classList)

        bestFeat = self.chooseBestFeature(D)
        bestFeatLabel = labels[bestFeat]
        featLabels = np.append(featLabels, bestFeatLabel)
        mytree = {bestFeatLabel: {}}
        labels = np.delete(labels, bestFeat)
        featValues = [feat[bestFeat] for feat in D]
        featValues = set(featValues)
        for value in featValues:
            mytree[bestFeatLabel][value] = self.create_tree(self.splitDataSet(D, bestFeat, value), labels, featLabels)

        return mytree

    def find_majority(self, pre):
        p = {}
        for son in list(pre.values()):
            if isinstance(son, dict):
                x, y = self.find_majority(son)
                p[x] = y
            else:
                if son in p:
                    p[son] += 1
                else:
                    p[son] = 1
        k, v = None, None
        for key, value in p.items():
            if k is None or value > v:
                k, v = key, value

        return k, v


    def get_tree(self):
        return self.mytree

    def predict(self, D):
        y = [0 for i in range(len(D))]
        for i in range(len(y)):
            a = D[i]
            pre = self.mytree
            while isinstance(pre, dict):
                key = list(pre.keys())[0]
                pre = pre[key]
                if a[key] in pre:
                    pre = pre[a[key]]
                else:
                    pre = self.find_majority(pre)
            y[i] = pre
        return y
