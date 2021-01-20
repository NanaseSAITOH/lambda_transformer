from torch.autograd import Variable
from learnning_model.Transformer import Transformer
import torch
from learnning_model.DataCollector import DataCollector
from learnning_model.SentenceFormatter import SentenceFormatter

import pickle

MAX_SEQ_LEN = 75
class PredictTaskExecutor:

    def decode_sentence(self, model, sentence, dataset, TEXT):
        model.eval()
        indexed = []
        for tok in sentence:
            if tok != torch.tensor(0, device='cuda:0' if torch.cuda.is_available() else "cpu"):
                indexed.append(tok)
            else:
                indexed.append(0)
        sentence = Variable(torch.LongTensor([indexed]))
        trg_init_tok = TEXT.vocab.stoi['<init>']
        trg = torch.LongTensor([[trg_init_tok]])
        translated_sentence = ""
        maxlen = MAX_SEQ_LEN
        for i in range(maxlen):
            size = trg.size(0)
            np_mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
            np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
            np_mask = np_mask
            pred = model(src=sentence.transpose(0, 1), tgt=trg, tgt_mask=np_mask)
            add_word = TEXT.vocab.itos[pred.argmax(dim=2)[-1].item()]
            translated_sentence += " " + add_word
            trg = torch.cat((trg, torch.LongTensor([[pred.argmax(dim=2)[-1]]])))
        return translated_sentence

    def main(self, sentence):
        dataset = DataCollector.load_data()

        encode_sentence = sentence

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open("/mnt/lambda/TEXT.pickle") as ff:
            TEXT = pickle.load(ff)

        # nn.LogSoftmax()を計算してからnn.NLLLoss(negative log likelihood loss)を計算
        model = Transformer(target_vocab_length＝TEXT.vocab.vectors.shape[0])

        model = model.cuda() if torch.cuda.is_available() else model.cpu()

        model_path = "/mnt/lambda/checkpoint_best_epoch_75_best.pt"

        model.load_state_dict(torch.load(
            model_path, map_location=torch.device('cpu')))

        mSentenceFormatter = SentenceFormatter()

        text_list = mSentenceFormatter.text_to_vector(
            texts=encode_sentence, TEXT=TEXT)

        text_tensor = torch.tensor(text_list).to(device)
        return self.decode_sentence(model, text_tensor, dataset,TEXT)
