import numpy as np

def Ent(D, l):
    p = {}
    for row in D:
        if row[l] in p:
            p[row[l]] += 1.0
        else:
            p[row[l]] = 1.0
    p = np.array(list(p.values()))
    p /= len(D)
    return -np.sum(p * np.log2(p))

def Gain_utils(ent, D, a, rho=1.0, dec=None, ratio=True):
    c = {}
    for row in D:
        if row[a] in c:
            c[row[a]].append(row)
        else:
            c[row[a]] = [row]

    gain = ent
    num = []
    for value in c.values():
        e = Ent(value, -1)
        gain -= e * len(value) / len(D)
        num.append(len(value) / len(D))
    gain *= rho
    IV = -np.sum(num * np.log2(num))
    gain_ratio = gain / IV

    if dec is not None:
        gain = round(gain, dec)
        gain_ratio = round(gain_ratio, dec)

    if ratio:
        return gain_ratio
    else:
        return gain

def IV(a, dec=None):
    p = {}
    for i in a:
        if i in p:
            p[i] += 1.0 / len(a)
        else:
            p[i] = 1.0 / len(a)
    p = np.array(list(p.values()))

    iv = -np.sum(p * np.log2(p))
    if dec is not None:
        iv = round(iv, dec)

    return iv
