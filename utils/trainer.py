import datetime
import os
import torch
import time
import re
from tqdm import tqdm



from abc import ABC, abstractmethod
from torch import nn

from tensorboard_logger import configure, log_value
import matplotlib.pyplot as plt

from utils.helpers import *


class Trainer(ABC):
    """
    Abstract base class which will serve as a NN trainer
    """
    def __init__(self, dataset,
                 model,
                 lr=1e-4):
        """
        Initializes the trainer class
        :param dataset: torch Dataset object
        :param model: torch.nn object
        :param lr: float, learning rate
        """
        self.dataset = dataset
        self.model = model
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )

    def train_model(self, batch_size, num_epochs, plot=False, log=False):
        """
        Trains the model
        :param batch_size: int,
        :param num_epochs: int,
        :param plot: bool, creates a matplotlib plot if TRUE
        :param log: bool, logs epoch stats for viewing in tensorboard if TRUE
        :return: None
        """
        # set-up plot and log parameters

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)  # 把模型移动到 GPU
        print(f"Using device: {device}")

        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d_%H-%M-%S")

        # 过滤非法字符，避免 Windows 路径错误
        # 获取模型名称，并过滤非法字符
        model_name = self.model.__class__.__name__  # 仅使用类名，避免路径过长
        dataset_name = getattr(self.dataset, "__class__", "UnknownDataset").__name__  # 获取数据集名称
        short_model_name = f"{model_name}_{dataset_name}"  # 组合模型和数据集信息

        # 确保目录名合法
        short_model_name = re.sub(r'[<>:"/\\|?*]', '_', short_model_name)

        # 组合 `log_dir`
        log_dir = os.path.join('runs', f"{short_model_name}_{st}")

        # 确保 `runs/` 目录存在
        os.makedirs(log_dir, exist_ok=True)

        print(f"Logging directory: {log_dir}")  # ✅ Debug 输出
        configure(log_dir)

        if plot:
            plot_parameters = self.plot_init()

        # get dataloaders
        generator_train, generator_val, _ = self.dataset.data_loaders(
        batch_size=batch_size,
        split=(0.70, 0.20)
        )

        print('Num Train Batches: ', len(generator_train))
        print('Num Valid Batches: ', len(generator_val))

        # train epochs
        for epoch_index in range(num_epochs):
            # update training scheduler
            self.update_scheduler(epoch_index)

            """
            # run training loop on training data
            self.model.train()
            mean_loss_train, mean_accuracy_train = self.loss_and_acc_on_epoch(
                data_loader=generator_train,
                epoch_num=epoch_index,
                train=True)

            # run evaluation loop on validation data
            self.model.eval()
            mean_loss_val, mean_accuracy_val = self.loss_and_acc_on_epoch(
                data_loader=generator_val,
                epoch_num=epoch_index,
                train=False)
            """
            # ✅ 训练模式 Train module
            self.model.train()
            mean_loss_train = 0
            mean_accuracy_train = 0

            for sample_id, batch in tqdm(enumerate(generator_train), desc=f"Epoch {epoch_index + 1}/{num_epochs}"):
                score_tensor, metadata_tensor = batch

                # ✅ 移动 batch 到 GPU move batch to GPU
                score_tensor = score_tensor.to(device).long()
                metadata_tensor = metadata_tensor.to(device).long()

                self.optimizer.zero_grad()

                # ✅ 计算损失 calculate loss
                loss, accuracy = self.loss_and_acc_for_batch(
                    (score_tensor, metadata_tensor), epoch_index, train=True
                )

                # ✅ 反向传播 backpropagation
                loss.backward()
                self.optimizer.step()

                # ✅ 计算平均损失 count average loss
                mean_loss_train += loss.mean().cpu().detach().numpy()
                if accuracy is not None:
                    mean_accuracy_train += accuracy.cpu().detach().numpy()

            mean_loss_train /= len(generator_train)
            mean_accuracy_train /= len(generator_train)

            # ✅ 评估模式 evaluate module
            self.model.eval()
            mean_loss_val = 0
            mean_accuracy_val = 0

            for sample_id, batch in tqdm(enumerate(generator_val), desc="Validation"):
                score_tensor, metadata_tensor = batch

                # ✅ 移动 batch 到 GPU move batch to GPU
                score_tensor = score_tensor.to(device)
                metadata_tensor = metadata_tensor.to(device)

                loss, accuracy = self.loss_and_acc_for_batch(
                    (score_tensor, metadata_tensor), epoch_index, train=False
                )

                mean_loss_val += loss.mean().cpu().detach().numpy()
                if accuracy is not None:
                    mean_accuracy_val += accuracy.cpu().detach().numpy()

            mean_loss_val /= len(generator_val)
            mean_accuracy_val /= len(generator_val)
            data_element = {
                'epoch_index': epoch_index,
                'num_epochs': num_epochs,
                'mean_loss_train': mean_loss_train,
                'mean_accuracy_train': mean_accuracy_train,
                'mean_loss_val': mean_loss_val,
                'mean_accuracy_val': mean_accuracy_val
            }
            # plot / log parameters
            if log:
                # log value in tensorboard for visualization
                log_value('train_loss', mean_loss_train, epoch_index)
                log_value('train_accu', mean_accuracy_train, epoch_index)
                log_value('valid_loss', mean_loss_val, epoch_index)
                log_value('valid_accu', mean_accuracy_val, epoch_index)
            if plot:
                self.plot_epoch_stats(
                    **plot_parameters,
                    **data_element,
                )

            # print epoch stats
            self.print_epoch_stats(**data_element)
            # save model
            self.model.save()

    def loss_and_acc_on_epoch(self, data_loader, epoch_num=None, train=True):
        """
        Computes the loss and accuracy for an epoch
        :param data_loader: torch dataloader object
        :param epoch_num: int, used to change training schedule
        :param train: bool, performs the backward pass and gradient descent if TRUE
        :return: loss values and accuracy percentages
        """
        mean_loss = 0
        mean_accuracy = 0
        for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            print(f"Processing batch {batch_idx + 1}/{len(data_loader)}")  # ✅ 进度打印

            # ✅ Debug: 确保 batch 不是 None
            if batch is None:
                print(f"❌ Error: Received an empty batch at index {batch_idx}")
                continue  # 跳过空 batch

            # ✅ Debug: 打印 batch 结构
            print(
                f"✅ Batch type: {type(batch)} | Batch length: {len(batch) if hasattr(batch, '__len__') else 'Unknown'}")

            # process batch data
            batch_data = self.process_batch_data(batch)

            if batch_data is None:
                print(f" Error: process_batch_data() returned None for batch index {batch_idx}")
                continue  # 跳过空 batch

            if not isinstance(batch_data, (tuple, list)) or len(batch_data) != 2:
                print(
                    f" Error: process_batch_data() did not return a tuple (score, metadata)! Instead got: {batch_data}")
                continue


            # zero the gradients
            self.zero_grad()

            # compute loss for batch
            try:
                loss, accuracy = self.loss_and_acc_for_batch(
                batch_data, epoch_num, train=train
                )
            except Exception as e:
                print(f"❌ Error: Exception occurred in loss_and_acc_for_batch() for batch {batch_idx}: {e}")
            continue  # 跳过异常 batch

            # compute backward and step if train
            if train:
                loss.backward()
                # self.plot_grad_flow()
                self.step()

            # compute mean loss and accuracy
            mean_loss += to_numpy(loss.mean())
            if accuracy is not None:
                mean_accuracy += to_numpy(accuracy)

        mean_loss /= len(data_loader)
        mean_accuracy /= len(data_loader)
        return (
            mean_loss,
            mean_accuracy
        )

    def zero_grad(self):
        """
        Zero the grad of the relevant optimizers
        :return:
        """
        self.optimizer.zero_grad()

    def step(self):
        """
        Perform the backward pass and step update for all optimizers
        :return:
        """
        self.optimizer.step()

    def plot_grad_flow(self):
        ave_grads = []
        layers = []
        for n, p in self.model.named_parameters():
            if p.requires_grad and ("bias" not in n):
                if p.grad is not None:
                    layers.append(n)
                    ave_grads.append(p.grad.abs().mean())
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient Flow")
        plt.grid(True)
        plt.show()

    @abstractmethod
    def loss_and_acc_for_batch(self, batch, epoch_num=None, train=True):
        """
        Computes the loss and accuracy for the batch
        Must return (loss, accuracy) as a tuple, accuracy can be None
        :param batch: torch Variable,
        :param epoch_num: int, used to change training schedule
        :param train: bool, True is backward pass is to be performed
        :return: scalar loss value, scalar accuracy value
        """
        pass

    @abstractmethod
    def process_batch_data(self, batch):
        """
        Processes the batch returned by the dataloader iterator
        :param batch: object returned by the dataloader iterator
        :return: torch Variable or tuple of torch Variable objects
        """
        pass

    @abstractmethod
    def update_scheduler(self, epoch_num):
        """
        Updates the training scheduler if any
        :param epoch_num: int,
        """
        pass

    @staticmethod
    def plot_init():
        fig, axarr = plt.subplots(2, sharex=True)
        fig.show()

        fig_parameters = {
            'fig': fig,
            'axarr': axarr,
            'x': [],
            'y_loss_train': [],
            'y_acc_train': [],
            'y_loss_val': [],
            'y_acc_val': [],
        }
        return fig_parameters

    @staticmethod
    def print_epoch_stats(epoch_index,
                          num_epochs,
                          mean_loss_train,
                          mean_accuracy_train,
                          mean_loss_val,
                          mean_accuracy_val):
        """
        Prints the epoch statistics
        :param epoch_index: int,
        :param num_epochs: int,
        :param mean_loss_train: float,
        :param mean_accuracy_train:float,
        :param mean_loss_val: float,
        :param mean_accuracy_val: float
        :return: None
        """
        print(
            f'Train Epoch: {epoch_index + 1}/{num_epochs}')
        string1 = f'Train Epoch: {epoch_index + 1}/{num_epochs}'
        print(f'\tTrain Loss: {mean_loss_train}'
              f'\tTrain Accuracy: {mean_accuracy_train * 100} %'
              )
        string2 = f'\tTrain Loss: {mean_loss_train}'
        string3 = f'\tTrain Accuracy: {mean_accuracy_train * 100} %'
        print(
            f'\tValid Loss: {mean_loss_val}'
            f'\tValid Accuracy: {mean_accuracy_val* 100} %'
        )
        string4 = f'\tValid Loss: {mean_loss_val}'
        string5 = f'\tValid Accuracy: {mean_accuracy_val* 100} %'

        file = open('summary.txt','a')
        file.write(string1)
        file.write('\n')
        file.write(string2)
        file.write('\n')
        file.write(string3)
        file.write('\n')
        file.write(string4)
        file.write('\n')
        file.write(string5)
        file.write('\n')
        file.write(str(datetime.datetime.now()))
        file.write('\n')
        file.close()


    @staticmethod
    def plot_epoch_stats(x, y_loss_train, y_acc_train,
                    y_loss_val, y_acc_val,
                    axarr, fig,
                    epoch_index,
                    num_epochs,
                    mean_loss_train,
                    mean_accuracy_train,
                    mean_loss_val,
                    mean_accuracy_val):
        x.append(epoch_index)
        y_loss_train.append(mean_loss_train)
        y_acc_train.append(mean_accuracy_train * 100)
        y_loss_val.append(mean_loss_val)
        y_acc_val.append(mean_accuracy_val * 100)
        axarr[0].plot(x, y_loss_train, 'r-', x, y_loss_val, 'r--')
        axarr[1].plot(x, y_acc_train, 'r-', x, y_acc_val, 'r--')
        fig.canvas.draw()
        plt.pause(0.001)

    @staticmethod
    def mean_crossentropy_loss(weights, targets):
        """
        Evaluates the cross entropy loss
        :param weights: torch Variable,
                (batch_size, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, seq_len)
        :return: float, loss
        """
        criteria = nn.CrossEntropyLoss(reduction='mean')
        batch_size, seq_len, num_notes = weights.size()
        assert (batch_size == targets.size(0))
        assert (seq_len == targets.size(1))
        weights = weights.view(-1, num_notes)
        targets = targets.view(-1)
        loss = criteria(weights, targets)
        return loss

    @staticmethod
    def mean_accuracy(weights, targets):
        """
        Evaluates the mean accuracy in prediction
        :param weights: torch Variable,
                (batch_size, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, seq_len)
        :return float, accuracy
        """
        _, _, num_notes = weights.size()
        weights = weights.view(-1, num_notes)
        targets = targets.view(-1)

        _, max_indices = weights.max(1)
        correct = max_indices == targets
        return torch.sum(correct.float()) / targets.size(0)

    @staticmethod
    def mean_l1_loss_rnn(weights, targets):
        """
        Evaluates the mean l1 loss
        :param weights: torch Variable,
                (batch_size, seq_len, hidden_size)
        :param targets: torch Variable
                (batch_size, seq_len, hidden_size)
        :return: float, l1 loss
        """
        criteria = nn.L1Loss()
        batch_size, seq_len, hidden_size = weights.size()
        assert (batch_size == targets.size(0))
        assert (seq_len == targets.size(1))
        assert (hidden_size == targets.size(2))
        loss = criteria(weights, targets)
        return loss

    @staticmethod
    def mean_mse_loss_rnn(weights, targets):
        """
        Evaluates the mean mse loss
        :param weights: torch Variable,
                (batch_size, seq_len, hidden_size)
        :param targets: torch Variable
                (batch_size, seq_len, hidden_size)
        :return: float, l1 loss
        """
        criteria = nn.MSELoss()
        batch_size, seq_len, hidden_size = weights.size()
        assert (batch_size == targets.size(0))
        assert (seq_len == targets.size(1))
        assert (hidden_size == targets.size(2))
        loss = criteria(weights, targets)
        return loss

    @staticmethod
    def mean_crossentropy_loss_alt(weights, targets):
        """
        Evaluates the cross entropy loss
        :param weights: torch Variable,
                (batch_size, num_measures, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, num_measures, seq_len)
        :return: float, loss
        """
        criteria = torch.nn.CrossEntropyLoss(reduction='mean')
        _, _, _, num_notes = weights.size()
        weights = weights.view(-1, num_notes)
        targets = targets.view(-1)
        loss = criteria(weights, targets)
        return loss

    @staticmethod
    def mean_accuracy_alt(weights, targets):
        """
        Evaluates the mean accuracy in prediction
        :param weights: torch Variable,
                (batch_size, num_measures, seq_len, num_notes)
        :param targets: torch Variable,
                (batch_size, num_measures, seq_len)
        :return float, accuracy
        """
        _, _, _, num_notes = weights.size()
        weights = weights.view(-1, num_notes)
        targets = targets.view(-1)
        _, max_indices = weights.max(1)
        correct = max_indices == targets
        return torch.sum(correct.float()) / targets.size(0)