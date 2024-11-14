import torch

from src.utils.constants import *

class Alphabet():
    def __init__(self, characters, mode='both'):
        self.mode = mode
        if mode == 'ctc':
            extra = [BLANK, PAD]
        elif mode == "attention":
            extra = [BLANK, PAD, START_OF_SEQUENCE, END_OF_SEQUENCE]
        elif mode == "both":
            extra = [BLANK, PAD, START_OF_SEQUENCE, END_OF_SEQUENCE]

        all_chars = [extra,characters]

        self.toPosition = {}
        self.toCharacter = {}
        self.labels = []
        id = 0
        for t in all_chars:
            for char in t:
                self.toPosition[char] = id
                self.toCharacter[id] = char
                id += 1
                self.labels.append(char)

    def check_valid_input_chars(self, text):
        for char in text:
            if char not in self.toPosition.keys():
                return False
        return True

    def return_wrong_input(self, text):
        wrong_chars = []
        for char in text:
            if char not in self.toPosition.keys():
                wrong_chars.append(char)
        return wrong_chars

    def string_to_logits(self, x_in):
        out = []
        for i in x_in:
            out.append(self.toPosition[i])
        return torch.LongTensor(out)

    def logits_to_string(self, x_in):
        out = []
        for i in x_in:
            out.append(self.toCharacter[int(i)])
        return "".join(out)

    def batch_logits_to_string_list(self, x_in, stopping_logits: list = None):
        text = []
        classification = []
        for b in x_in:
            if stopping_logits is None:
                text.append(self.logits_to_string(b))
                classification.append(torch.Tensor([self.toPosition[PAD]]))
            else:
                stops = []
                for s in stopping_logits:
                    stop = torch.where(b == s)[0]
                    if len(stop) == 0:
                        stop = torch.LongTensor([len(b)])
                    stops.append(stop[0])
                end_idx = torch.min(torch.stack(stops))
                text.append(self.logits_to_string(b[:end_idx]))
                if end_idx == len(b):
                    classification.append(torch.Tensor([self.toPosition[PAD]]))
                else:
                    end_classifier = torch.argmin(torch.stack(stops))
                    classification.append(torch.Tensor([stopping_logits[end_classifier]]))
        return text, torch.stack(classification)
