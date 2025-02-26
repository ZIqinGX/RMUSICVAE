import torch
import torch.nn.functional as F
from torch import distributions
from utils.trainer import Trainer
from MeasureVAE.measure_vae import MeasureVAE
from data.dataloaders.bar_dataset import *


class VAETrainer(Trainer):
    def __init__(
        self, dataset,
        model: MeasureVAE,
        lr=1e-4,
        has_reg_loss=False,
        reg_type=None,
        reg_dim=0,
    ):
        super(VAETrainer, self).__init__(dataset, model, lr)
        self.beta = 0.001
        self.cur_epoch_num = 0
        self.has_reg_loss = has_reg_loss
        if self.has_reg_loss:
            self.reg_type = 'rhy_complexity'  # 强制只用 rhythmic complexity
            self.reg_dim = 0  # 只对齐 z0
            self.trainer_config = '[' + self.reg_type + ',' + str(self.reg_dim) + ']'
            self.model.update_trainer_config(self.trainer_config)
        self.warm_up_epochs = 10

    def process_batch_data(self, batch):
        """
        Processes the batch returned by the dataloader iterator
        :param batch: object returned by the dataloader iterator
        :return: tuple of Torch Tensor objects (score, metadata)
        """
        print(f"📦 process_batch_data() received batch of type: {type(batch)}")

        if batch is None:
            print("❌ Error: batch is None!")
            raise ValueError("Batch is None!")

        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            print(f"❌ Error: Invalid batch structure! Expected (score, metadata), but got: {batch}")
            raise ValueError("Invalid batch structure! Expected (score, metadata)")

        score_tensor, metadata_tensor = batch

        if not isinstance(score_tensor, torch.Tensor) or not isinstance(metadata_tensor, torch.Tensor):
            print("❌ Error: Expected both score_tensor and metadata_tensor to be torch.Tensor")
            raise TypeError("Both score and metadata must be torch.Tensor")

        device = next(self.model.parameters()).device
        score_tensor = score_tensor.to(device).long()
        metadata_tensor = metadata_tensor.to(device).long()

        return score_tensor, metadata_tensor

    def loss_and_acc_for_batch(self, batch, epoch_num=None, train=True):
        """
        Computes the loss and accuracy for the batch
        :param batch: tuple (score, metadata)
        :param epoch_num: int, epoch index
        :param train: bool, True if backward pass should be performed
        :return: scalar loss value, scalar accuracy value
        """
        try:
            score, metadata = self.process_batch_data(batch)
        except Exception as e:
            print(f"❌ Error processing batch: {e}")
            return None, None

        print(f"Score device: {score.device}, Metadata device: {metadata.device}")

        if self.cur_epoch_num != epoch_num:
            flag = True
            self.cur_epoch_num = epoch_num
        else:
            flag = False

        weights, samples, z_dist, prior_dist, z_tilde, z_prior = self.model(
            measure_score_tensor=score,
            measure_metadata_tensor=metadata,
            train=train
        )

        recons_loss = self.mean_crossentropy_loss(weights=weights, targets=score.long())
        dist_loss = self.compute_kld_loss(z_dist, prior_dist)
        loss = recons_loss + dist_loss

        accuracy = self.mean_accuracy(weights=weights, targets=score)

        if self.has_reg_loss:
            reg_loss = self.compute_reg_loss(z_tilde, score)
            loss += reg_loss
            if flag:
                print(recons_loss.item(), dist_loss.item(), reg_loss.item())
        else:
            if flag:
                print(recons_loss.item(), dist_loss.item())

        return loss, accuracy

    def update_scheduler(self, epoch_num):
        gamma = 0.00495
        if not self.has_reg_loss:
            if self.warm_up_epochs < epoch_num < 31:
                self.beta += gamma
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']
            break
        print('LR: ', current_lr, ' Beta: ', self.beta)

    def step(self):
        self.optimizer.step()

    def compute_reg_loss(self, z, score):
        attr_tensor = self.dataset.get_rhy_complexity(score).long()
        x = z[:, 0]
        reg_loss = self.reg_loss_sign(x, attr_tensor)
        return reg_loss

    def compute_kld_loss(self, z_dist, prior_dist):
        kld = distributions.kl.kl_divergence(z_dist, prior_dist)
        kld = self.beta * kld.sum(1).mean()
        return kld

    @staticmethod
    def reg_loss_sign(x, y):
        x = x.view(-1, 1).repeat(1, x.shape[0])
        x_diff_sign = (x - x.transpose(1, 0)).view(-1, 1)
        x_diff_sign = torch.tanh(x_diff_sign * 10)

        y = y.view(-1, 1).repeat(1, y.shape[0])
        y_diff_sign = torch.sign(y - y.transpose(1, 0)).view(-1, 1)

        loss_fn = torch.nn.L1Loss()
        sign_loss = loss_fn(x_diff_sign, y_diff_sign)
        return sign_loss

    @staticmethod
    def compute_kernel(x, y, k):
        batch_size_x, dim_x = x.size()
        batch_size_y, dim_y = y.size()
        assert dim_x == dim_y

        xx = x.unsqueeze(1).expand(batch_size_x, batch_size_y, dim_x)
        yy = y.unsqueeze(0).expand(batch_size_x, batch_size_y, dim_y)
        distances = (xx - yy).pow(2).sum(2)
        return k(distances)

    @staticmethod
    def compute_mmd_loss(z_tilde, z_prior, coeff=10):
        def gaussian(d, var=16.):
            return torch.exp(- d / var).sum(1).sum(0)

        k = gaussian
        batch_size = z_tilde.size(0)
        zp_ker = VAETrainer.compute_kernel(z_prior, z_prior, k)
        zt_ker = VAETrainer.compute_kernel(z_tilde, z_tilde, k)
        zp_zt_ker = VAETrainer.compute_kernel(z_prior, z_tilde, k)

        first_coeff = 1. / (batch_size * (batch_size - 1)) / 2 if batch_size > 1 else 1
        second_coeff = 2 / (batch_size * batch_size)
        mmd = coeff * (first_coeff * zp_ker + first_coeff * zt_ker - second_coeff * zp_zt_ker)
        return mmd
