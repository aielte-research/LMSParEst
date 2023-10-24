'''
the metric block class "glues" metrics together and handles the methods
without having to write loops in the main part of the code
'''

class MetricBlock():
    def __init__(self, class_list, train):
        self.metric_list = [c(train = train) for c in class_list]
        
    def get_final(self):
        for m in self.metric_list:
            m.get_final()

    def update(self, batch_input, batch_label):
        for m in self.metric_list:
            m.update(batch_input, batch_label)

    def epoch_log(self):
        for m in self.metric_list:
            m.epoch_log()

    def batch_log(self):
        for m in self.metric_list:
            m.batch_log()

    def __str__(self):
        return '\n'.join(str(m) for m in self.metric_list)


