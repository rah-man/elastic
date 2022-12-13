import numpy as np

class Metrics:
    def __init__(self):
        self.accuracy = {}
        self.forget = {}

    def add_accuracy(self, taskid, acc):
        cur = self.accuracy.get(taskid, [])
        cur.append(acc)
        self.accuracy[taskid] = cur

    def final_average_accuracy(self):
        return self.average_accuracy(len(self.accuracy)-1)

    def average_accuracy(self, taskid):
        # FAA is calculated on self.accuracy[len(self.accuracy)-1]
        return np.average(self.accuracy[taskid])

    def add_forgetting(self, taskid):
        temp_ = {}
        for cur in range(taskid):
            for each in range(cur+1):                
                # print(each, cur, taskid, sep=", ")
                diff = self.accuracy[cur][each] - self.accuracy[taskid][each]
                t = temp_.get(each, [])
                t.append(diff)
                temp_[each] = t

        for i in range(len(temp_)):
            temp_[i] = max(temp_[i])
        self.forget[taskid] = temp_

    def forgetting(self, taskid):
        # FF is calculated on self.forget[len(self.forget)-1]
        if taskid > 0:
            return sum([v for k, v in self.forget[taskid].items()]) / len(self.forget[taskid])
        return 0

    def final_forgetting(self):
        return self.forgetting(len(self.forget)-1)

if __name__ == "__main__":
    met = Metrics()
    met.add_accuracy(0, 0.98)
    met.add_accuracy(1, 0.53)
    met.add_accuracy(1, 0.97)
    met.add_accuracy(2, 0.31)
    met.add_accuracy(2, 0.59)
    met.add_accuracy(2, 0.96)
    met.add_accuracy(3, 0.11)
    met.add_accuracy(3, 0.29)
    met.add_accuracy(3, 0.51)
    met.add_accuracy(3, 0.97)
    met.add_accuracy(4, 0.05)
    met.add_accuracy(4, 0.11)
    met.add_accuracy(4, 0.21)
    met.add_accuracy(4, 0.55)
    met.add_accuracy(4, 0.85)

    for i in range(len(met.accuracy)):
        print(met.accuracy[i])
    print()

    met.forgetting(0)
    met.forgetting(1)
    met.forgetting(2)
    met.forgetting(3)
    met.forgetting(4)
    
    for k, v in met.forget.items():
        print(v)
        print()