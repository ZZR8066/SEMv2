from collections import defaultdict
import time
import datetime


class TimeCounter:
    def __init__(self, start_epoch, num_epochs, epoch_iters):
        self.start_epoch = start_epoch
        self.num_epochs = num_epochs
        self.epoch_iters = epoch_iters
        self.start_time = None

    def reset(self):
        self.start_time = time.time()

    def step(self, epoch, batch):
        used = time.time() - self.start_time
        finished_batch_nums = (epoch - self.start_epoch) * self.epoch_iters + batch
        batch_time_cost = used / finished_batch_nums
        total = (self.num_epochs - self.start_epoch) * self.epoch_iters * batch_time_cost
        left = total - used
        return str(datetime.timedelta(seconds=left))


def format_table(table, padding=1):
    table = [[str(subitem) for subitem in item] for item in table]
    num_cols = max([len(item) for item in table])
    cols_width = [0] * num_cols

    for row in table:
        for col_idx, cell in enumerate(row):
            cols_width[col_idx] = max(cols_width[col_idx], len(cell))

    string = '┌'
    for col_idx in range(num_cols):
        string += '─' * (padding * 2 + cols_width[col_idx])
        if col_idx == num_cols - 1:
            string += '┐'
        else:
            string += '┬'
    string += '\n'

    for row_idx, row in enumerate(table):
        string += '│'
        for col_idx in range(num_cols):
            if col_idx < len(row):
                word = row[col_idx]
            else:
                word = ''
            col_width = cols_width[col_idx]
            left_pad = (col_width - len(word))//2
            right_pad = col_width - len(word) - left_pad
            string += ' ' * (padding + left_pad)
            string += word
            string += ' ' * (padding + right_pad)
            string += '│'
        
        string += '\n'

        if row_idx < len(table) - 1:
            string += '├'
        else:
            string += '└'
        for col_idx in range(num_cols):
            string += '─' * (padding * 2 + cols_width[col_idx])
            if col_idx == num_cols - 1:
                if row_idx < len(table) - 1:
                    string += '┤'
                else:
                    string += '┘'
            else:
                if row_idx < len(table) - 1:
                    string += '┼'
                else:
                    string += '┴'

        string += '\n'
    return string


class TicTocCounter:
    def __init__(self):
        self.tics = dict()
        self.seps = defaultdict(list)

    def tic(self, name):
        self.tics[name] = time.time()
    
    def toc(self, name):
        toc = time.time()
        if name in self.tics:
            self.seps[name].append(toc-self.tics[name])
    
    def __repr__(self):
        string = 'TicTocCount Result:\n'
        infos = [['Name', 'Mean Time', 'Total Time']]
        for key, val in self.seps.items():
            mean = sum(val)/len(val)
            total = sum(val)
            infos.append([key, '%0.4f' % mean, '%0.4f' % total])
        string += format_table(infos)
        return string

    def reset(self):
        self.tics.clear()
        self.seps.clear()

global_tictoc_counter = TicTocCounter()
