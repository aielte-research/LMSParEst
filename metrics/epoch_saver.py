class Epoch_saver():
    def __init__(self, params, neptune_experiment):
        self.__dict__ = params

    def __call__(self, epoch_res):
        
        if hasattr(self, 'start_count'):
            if self.start_count == 'last_ep':
                self.start_count = self.num_epochs

            to_be_saved = self.curr_epoch >= self.start_count

        else:
            to_be_saved = True

        if to_be_saved:
            return [([a[0] for a in epoch_res], [a[1] for a in epoch_res])]

        else:
            return []