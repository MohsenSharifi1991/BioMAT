import time
import numpy as np
import torch
import wandb

from model.lstmlstm import Seq2SeqTest
from model.lstmlstmattention import Seq2SeqAttTest
from model.transformer_seq2seq import Seq2SeqTransformerTest
from modelbuilder import ModelBuilder


class Test:
    def run_testing(self, config, model, test_dataloader):
        self.config = config
        self.device = config['device']
        self.loss = self.config['loss']
        self.weight = self.config['loss_weight']
        self.model_name = self.config['model_name']
        self.classification = config['classification']
        self.n_output = len(self.config['selected_opensim_labels'])
        if not self.n_output == len(self.weight):
            self.weight = None
        modelbuilder_handler = ModelBuilder(self.config)
        criterion = modelbuilder_handler.get_criterion(self.weight)
        self.tester = self.setup_tester()
        y_pred, y_true, loss = self.tester(model, test_dataloader, criterion, self.device)
        return y_pred, y_true, loss

    def setup_tester(self):
        if self.model_name == 'seq2seqatt':
            tester = self.testing_seq2seqatt
        elif self.model_name == 'seq2seqtransformer':
            tester = self.testing_transformer_seq2seq
        elif (self.model_name == 'transformer' and not self.classification) or (self.model_name == 'transformertsai' and not self.classification):
            tester = self.testing_transformer
        elif self.classification:
            tester = self.testing_w_classification
        else:
            tester = self.testing
        return tester

    def testing(self, model, test_dataloader, criterion, device):
        model.eval()
        with torch.no_grad():
            test_loss = []
            test_preds = []
            test_trues = []
            inference_times = []
            for x, y in test_dataloader:
                x = x.to(device)
                y = y.to(device)
                start_time = time.time()
                y_pred = model(x.float())
                inference_times.append(time.time() - start_time)
                loss = criterion(y, y_pred)
                test_loss.append(loss.item())
                test_preds.append(y_pred)
                test_trues.append(y)
            test_loss = torch.mean(torch.tensor(test_loss))
            print('Test Accuracy of the model: {}'.format(test_loss))
        # wandb.log({"inference time": np.mean(np.array(inference_times))})
        # wandb.log({"Test Loss": test_loss})
        return torch.cat(test_preds, 0), torch.cat(test_trues, 0), test_loss

    def testing_w_classification(self, model, test_dataloader, criterion, device):
        model.eval()
        with torch.no_grad():
            test_loss = []
            test_preds = []
            test_trues = []
            for x, y, y_label in test_dataloader:
                x = x.to(device).float()
                y_label = y_label.type(torch.LongTensor).to(device)  # The targets passed to nn.CrossEntropyLoss() should be in torch.long format
                y = y.to(device)
                y_pred = model(x)
                y_pred[0] = y_pred[0].double()
                y_pred[1] = y_pred[1].double()
                y_true = [y, y_label]
                loss = criterion(y_pred, y_true)
                test_loss.append(loss.item())
                test_preds.append(y_pred)
                test_trues.append(y_true)
            test_loss = torch.mean(torch.tensor(test_loss))
            print('Test Accuracy of the model: {}'.format(test_loss))
        wandb.log({"Test Loss": test_loss})
        test_preds_reg = []
        test_trues_reg = []
        for pred, true in zip(test_preds, test_trues):
            test_preds_reg.append(pred[0])
            test_trues_reg.append(true[0])
        return torch.cat(test_preds_reg, 0), torch.cat(test_trues_reg, 0), test_loss

    def testing_seq2seq(self, model, test_dataloader, criterion, device):
        model.eval()
        with torch.no_grad():
            test_loss = []
            test_preds = []
            test_trues = []
            for x, y in test_dataloader:
                x = x.to(device)
                y = y.to(device)
                # y_pred = model(x.float(), y.float())  # just for seq 2 seq
                y_pred = Seq2SeqTest(model, x.float())
                loss = criterion(y_pred[:, 1:, :].to(device), y[:, 1:, :])
                test_loss.append(loss.item())
                test_preds.append(y_pred)
                test_trues.append(y)
            test_loss = torch.mean(torch.tensor(test_loss))
            print('Test Accuracy of the model: {}'.format(test_loss))
        wandb.log({"Test Loss": test_loss})
        return torch.cat(test_preds, 0), torch.cat(test_trues, 0), test_loss

    def testing_seq2seqatt(self, model, test_dataloader, criterion, device):
        model.eval()
        with torch.no_grad():
            test_loss = []
            test_preds = []
            test_trues = []
            for x, y in test_dataloader:
                x = x.to(device)
                y = y.to(device)
                # y_pred = model(x.float(), y.float())  # just for seq 2 seq
                y_pred = Seq2SeqAttTest(model, x.float())
                loss = criterion(y_pred[:, 1:, :].to(device), y[:, 1:, :])
                test_loss.append(loss.item())
                test_preds.append(y_pred)
                test_trues.append(y)
            test_loss = torch.mean(torch.tensor(test_loss))
            print('Test Accuracy of the model: {}'.format(test_loss))
        wandb.log({"Test Loss": test_loss})
        return torch.cat(test_preds, 0), torch.cat(test_trues, 0), test_loss

    def testing_transformer(self, model, test_dataloader, criterion, device):
        model.eval()
        with torch.no_grad():
            test_loss = []
            test_preds = []
            test_trues = []
            inference_times = []
            for x, y in test_dataloader:
                x = x.to(device)
                y = y.to(device)
                # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start_time = time.time()
                y_pred = model(x.float())  # just for transformer
                inference_times.append(time.time()-start_time)
                loss = criterion(y, y_pred.to(device))
                test_loss.append(loss.item())
                test_preds.append(y_pred)
                test_trues.append(y)
            test_loss = torch.mean(torch.tensor(test_loss))
            print('Test Accuracy of the model: {}'.format(test_loss))
        # wandb.log({"Test Loss": test_loss})
        # wandb.log({"inference time": np.mean(np.array(inference_times))})
        return torch.cat(test_preds, 0), torch.cat(test_trues, 0), test_loss

    def testing_transformer_seq2seq(self, model, test_dataloader, criterion, device):
        model.eval()
        with torch.no_grad():
            test_loss = []
            test_preds = []
            test_trues = []
            for x, y in test_dataloader:
                x = x.to(device)
                y = y.to(device)
                y_pred = Seq2SeqTransformerTest(model, x.float())
                # y_pred = model(x.float(), y.float()[:, :-1, :])  # just for seq 2 seq transformer
                loss = criterion(y_pred, y.to(device))
                test_loss.append(loss.item())
                test_preds.append(y_pred)
                test_trues.append(y[:, 1:, :])
            test_loss = torch.mean(torch.tensor(test_loss))
            print('Test Accuracy of the model: {}'.format(test_loss))
        # wandb.log({"Test Loss": test_loss})
        return torch.cat(test_preds, 0), torch.cat(test_trues, 0), test_loss