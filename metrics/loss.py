class Loss():
    def __init__(self, loss_fun, train):
        self.loss = 0
        self.loss_fun = loss_fun
        self.train = train

    def update(self, batch_label, inferred_label):
        self.new_loss = self.loss_fun(batch_label, inferred_label)
        self.loss += self.new_loss

    def backward(self):
        self.new_loss.backward()

    def epoch_log(self):
        pass

    def batch_log(self):
        pass
    
    def get_final(self, batch_count):
        self.loss = self.loss / batch_count

    def __str__(self):
        k = ('train' if self.train else 'val') + '_' + 'loss'
        v = self.loss.item()

        return str({k: v})
