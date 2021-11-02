import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from rnn_cell import RNNCell, GRUCell, LSTMCell

class RNN(nn.Module):
    def __init__(self,
            num_embed_units,  # pretrained wordvec size
            num_units,        # RNN units size
            num_vocabs,       # vocabulary size
            wordvec,          # pretrained wordvec matrix
            dataloader,       # dataloader
            type = 0,         # 0 for RNN, 1 for GRU, 2 for LSTM
        ): 
        super().__init__()

        # load pretrained wordvec
        self.wordvec = nn.Embedding.from_pretrained(wordvec)
        # the dataloader
        self.dataloader = dataloader

        # TODO START
        # fill the initialization of cells
        self.type = type
        if type == 0:
            self.cell = RNNCell(num_embed_units, num_units)
        elif type == 1:
                self.cell = GRUCell(num_embed_units, num_units)
        elif type == 2:
                self.cell = LSTMCell(num_embed_units, num_units)
        else:
            raise NotImplementedError("Wrong type")
        # TODO END

        # intialize other layers
        self.linear = nn.Linear(num_units, num_vocabs)

    def forward(self, batched_data, device):
        # Padded Sentences
        sent = torch.tensor(batched_data["sent"], dtype=torch.long, device=device) # shape: (batch_size, length)
        # An example:
        #   [
        #   [2, 4, 5, 6, 3, 0],   # first sentence: <go> how are you <eos> <pad>
        #   [2, 7, 3, 0, 0, 0],   # second sentence:  <go> hello <eos> <pad> <pad> <pad>
        #   [2, 7, 8, 1, 1, 3]    # third sentence: <go> hello i <unk> <unk> <eos>
        #   ]
        # You can use self.dataloader.convert_ids_to_sentence(sent[0]) to translate the first sentence to string in this batch.

        # Sentence Lengths
        length = torch.tensor(batched_data["sent_length"], dtype=torch.long, device=device) # shape: (batch)
        # An example (corresponding to the above 3 sentences):
        #   [5, 3, 6]

        batch_size, seqlen = sent.shape

        # TODO START
        # implement embedding layer
        embedding = self.wordvec(sent)# shape: (batch_size, length, num_embed_units)
        # TODO END

        now_state = self.cell.init(batch_size, device)

        loss = 0
        logits_per_step = []
        for i in range(seqlen - 1):
            hidden = embedding[:, i]
            hidden, now_state = self.cell(hidden, now_state) # shape: (batch_size, num_units)
            logits = self.linear(hidden) # shape: (batch_size, num_vocabs)
            logits_per_step.append(logits)

        # TODO START
        # calculate loss
        loss = 0
        for step in range(len(logits_per_step)):
            front = sent[:, step + 1]
            logits = logits_per_step[step]
            mask = (front != 0).view(-1, 1)
            logits = F.softmax(logits, dim=1)
            logits = -torch.log(logits.gather(1, front.unsqueeze(1)))
            loss += logits.masked_select(mask).sum()
        loss /= length.sum().item()
        loss = loss.to(device)
        # TODO END

        return loss, torch.stack(logits_per_step, dim=1)
    def top_p_sampler(self, logits, margin = 1):
        # Reference: https://gongel.cn/?p=7119
        filter_value = -1e10
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=1)
        cumsum = torch.cumsum(F.softmax(sorted_logits, dim=1), dim=1)
        to_remove = cumsum > margin
        to_remove[...,1:] = to_remove[..., :-1].clone()
        to_remove[..., 0] = 0
        batch_size = logits.size(0)
        for i in range(batch_size):
            removed = sorted_indices[i, to_remove[i]]
            logits[i, removed] = filter_value
        return logits
    def inference(self, batch_size, device, decode_strategy, temperature, max_probability):
        # First Tokens is <go>
        now_token = torch.tensor([self.dataloader.go_id] * batch_size, dtype=torch.long, device=device)
        flag = torch.tensor([1] * batch_size, dtype=torch.float, device=device)

        now_state = self.cell.init(batch_size, device)

        generated_tokens = []
        for _ in range(50): # max sentecne length

            # TODO START
            # translate now_token to embedding
            embedding = self.wordvec(now_token)# shape: (batch_size, num_embed_units)
            # TODO END

            hidden = embedding
            hidden, now_state = self.cell(hidden, now_state)
            logits = self.linear(hidden) # shape: (batch_size, num_vocabs)

            if decode_strategy == "random":
                prob = (logits / temperature).softmax(dim=-1) # shape: (batch_size, num_vocabs)
                now_token = torch.multinomial(prob, 1)[:, 0] # shape: (batch_size)
            elif decode_strategy == "top-p":
                # TODO START
                # implement top-p samplings
                prob = (self.top_p_sampler(logits, margin=max_probability)).softmax(dim=-1)
                now_token = torch.multinomial(prob, 1)[:, 0] # shape: (batch_size)
                # TODO END
            else:
                raise NotImplementedError("unknown decode strategy")

            generated_tokens.append(now_token)
            flag = flag * (now_token != self.dataloader.eos_id)

            if flag.sum().tolist() == 0: # all sequences has generated the <eos> token
                break

        return torch.stack(generated_tokens, dim=1).detach().cpu().numpy()
