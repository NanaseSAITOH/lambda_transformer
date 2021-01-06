from torch.autograd import Variable
from learnning_model.Transformer import Transformer
import torch
from learnning_model.DataCollector import DataCollector
from learnning_model.SentenceFormatter import SentenceFormatter


class PredictTaskExecutor:

    def decode_sentence(self, model, sentence, dataset):
        indices_word = dataset['indices2word']
        word_indices = dataset['word2indices']
        model.eval()
        indexed = []
        for tok in sentence:
            if tok != torch.tensor(0, device='cuda:0' if torch.cuda.is_available() else "cpu"):
                indexed.append(tok)
            else:
                indexed.append(0)
        sentence = Variable(torch.LongTensor([indexed]))
        BOS_WORD = 'RESRES'
        trg_init_tok = word_indices[BOS_WORD]
        trg = torch.LongTensor([[trg_init_tok]])
        translated_sentence = ""
        maxlen = 74
        for i in range(maxlen):
            size = trg.size(0)
            np_mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
            np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')
                                                  ).masked_fill(np_mask == 1, float(0.0))
            np_mask = np_mask
            pred = model(src=sentence.transpose(
                0, 1), tgt=trg, tgt_mask=np_mask)
            add_word = indices_word[pred.argmax(dim=2)[-1].item()]
            translated_sentence += " " + add_word
            trg = torch.cat((trg, torch.LongTensor(
                [[pred.argmax(dim=2)[-1]]])))
        # print(trg)
        return translated_sentence

    def main(self, sentence):
        dataset = DataCollector.load_data()

        encode_sentence = sentence

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 損失関数の設定
        criterion = nn.CrossEntropyLoss()
        # nn.LogSoftmax()を計算してからnn.NLLLoss(negative log likelihood loss)を計算
        model = Transformer(words_num=len(dataset['words']))
        model = model.cuda() if torch.cuda.is_available() else model.cpu()
        model_path = "/mnt/lambda/checkpoint_best_epoch_75_best.pt"
        model.load_state_dict(torch.load(
            model_path, map_location=torch.device('cpu')))
        mSentenceFormatter = SentenceFormatter()
        text_list = mSentenceFormatter.text_to_vector(
            texts=encode_sentence, datasets=dataset)
        text_tensor = torch.tensor(text_list).to(device)
        return self.decode_sentence(model, text_tensor, dataset)
