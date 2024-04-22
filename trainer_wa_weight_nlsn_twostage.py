import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn as nn
from model import fsrcnn, edsr_baseline, rcan, swinir, nlsn
import numpy as np


def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	# print(net)
	print('Total number of parameters: %d' % num_params)


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("!!!!!!!!!!!!!!")
    print('Total', total_num, 'Trainable', trainable_num)
    print("!!!!!!!!!!!!!!")


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        # self.loader_train = loader.loader_train
        self.loader_train_normal = loader.loader_train_normal
        self.loader_train_inverse = loader.loader_train_inverse
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss

        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        self.criterion = nn.L1Loss()

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

        # calculate the model parameters
        get_parameter_number(self.model)

    def search_filter_index(self, sorted_filter_location, epoch):

        # sorted_noisy_location
        indexs = sorted_filter_location.tolist()#
        remain_factor = (epoch / self.args.epochs)*0.1 # 0.5
        N = round(remain_factor * len(indexs))
        final_index = torch.tensor(indexs[0:N]).cuda()  #

        # generate the remaining masks
        mask_flatten = torch.ones(len(indexs)).cuda()  # 576
        mask_flatten[final_index] = mask_flatten[final_index, ...] * 0
        self.mask = []
        self.map = []
        weight_num = 0
        pre = 0
        self.old_weight = []
        for k, w in self.model.named_parameters():
            if k.find('weight') != -1 and k.find('mean') == -1:
                if len(w.shape) == 4:
                    weight_num += len(w.view(-1))
                    each_layer = mask_flatten[pre:weight_num].view(w.shape[0], w.shape[1], w.shape[2], w.shape[3])
                    self.mask.append(each_layer)
                    inverse_each_layer = torch.ones_like(each_layer).cuda() - each_layer
                    self.map.append(inverse_each_layer)
                    old_weight = w * inverse_each_layer
                    self.old_weight.append(old_weight)
                    pre = weight_num

    def change_weight(self):
        i = 0
        for k, p in self.model.named_parameters():
            if k.find('weight') != -1 and k.find('mean') == -1:
                if len(p.shape) == 4:
                    p.data.mul_(self.mask[i]).add_(self.old_weight[i])
                    i = i + 1

    def change_filter(self):
        i = 0
        for k, v in self.model.named_parameters():
            if k.find('weight') != -1 and k.find('mean') == -1:
                v.data.mul_(self.mask[i].view(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1)).add_(self.old_filter[i])
                i = i + 1

    def faig(self, img1, gt_img, baseline_model_path, target_model_path, total_step, conv_name_list):
        """ Parameter Attribution Integrated Gradients.
            Reference From: https://github.com/TencentARC/FAIG
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        baseline_state_dict = torch.load(baseline_model_path)  # ['params_ema']
        target_state_dict = torch.load(target_model_path)  # ['params_ema']

        # calculate the gradient of two images with different degradation
        total_gradient_img1 = 0
        # approximate the integral via 100 discrete points uniformly
        # sampled along the straight-line path
        for step in range(0, total_step):
            # define current net
            alpha = step / total_step
            current_net_state_dict = {}
            for key, _ in baseline_state_dict.items():
                # a straight-line path between baseline model and target model
                current_net_state_dict[key] = alpha * baseline_state_dict[key] + (1 - alpha) * target_state_dict[key]

            # current_net = MSRResNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=scale)
            current_net = nlsn.NLSN(self.args)
            # current_net = rcan.RCAN(self.args)
            current_net.eval()
            current_net = current_net.to(device)
            current_net.load_state_dict(current_net_state_dict)

            # for degradation 1
            current_net.zero_grad()
            output1 = current_net(img1)
            # measure the distance between the network output and the ground-truth
            criterion = torch.nn.L1Loss(reduction='mean')
            loss1 = criterion(gt_img, output1)
            # calculate the gradient of F to every filter
            loss1.backward()
            # save the gradient of all filters to grad_list_img1
            grad_list_img1 = []

            # for i, w in enumerate(current_net.parameters()):
            # for key, value in current_net.items():
            for key, w in current_net.named_parameters():
                if key.find('weight') != -1 and key.find('mean') == -1:
                    # for nlsn
                    if len(w.shape) == 4:
                        grad = w.grad
                        grad = grad.reshape(-1) # , grad.shape[2], grad.shape[3]
                        # grad = grad.reshape(-1, 3, 3)
                        grad_list_img1.append(grad)

            # reshape to [-1, 3, 3]
            grad_list_img1 = torch.cat(grad_list_img1, dim=0)
            total_gradient_img1 += grad_list_img1

        # multiple the variation
        diff_list = []
        for key in conv_name_list:
            variation = baseline_state_dict[key] - target_state_dict[key]
            variation = variation.reshape(-1)  # , 3, 3)
            # variation = variation.reshape(-1, 3, 3)
            diff_list.append(variation)
        diff_list = torch.cat(diff_list, dim=0).to(device)

        single_faig_img1 = abs(total_gradient_img1 * diff_list / total_step)

        return single_faig_img1.cpu().numpy()

    def extract_index(self, epoch, target_model_path, baseline_model_path):
        timer_data, timer_model = utility.timer(), utility.timer()

        faig_average_noisy = 0.0
        print('Now we sort the filters for noise!')
        record_filters_folder = self.args.record_filters_folder
        os.makedirs(record_filters_folder, exist_ok=True)

        conv_name_list = []
        cumulate_num_neurons = [0]

        target_state_dict = torch.load(target_model_path)

        # Note that we exclude bias
        for key, value in target_state_dict.items():
            if key.find('weight') != -1 and key.find('mean') == -1:
                if len(value.shape) == 4:
                    conv_name_list.append(key)
                    num_neurons = value.size(0) * value.size(1) * value.size(2) * value.size(3)
                    cumulate_num_neurons.append(cumulate_num_neurons[-1] + num_neurons)
        # del the first element in cumulate_num_neurons
        del cumulate_num_neurons[0]

        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train_normal):
            timer_data.hold()

            lr, hr = self.prepare([lr, hr])

            timer_data.hold()
            timer_model.tic()

            # calculate the neurons
            diff = self.faig(
                lr,
                hr,
                baseline_model_path,
                target_model_path,
                self.args.total_step,
                conv_name_list)
            faig_average_noisy += np.array(diff)

            timer_model.hold()
            if (batch + 1) % self.args.print_every == 0:
                # print(batch) self.ckp.write_log
                print('[{}/{}]\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train_normal.dataset),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        # sort the neurons in descending order
        sorted_noisy_location = np.argsort(faig_average_noisy)[::-1]  # from big to small
        save_noisy_filter_txt = os.path.join(self.args.record_filters_folder,
                                             'IG_index_weight_un'+str(self.args.un_factor)+str(epoch) + '.txt')
        np.savetxt(save_noisy_filter_txt, sorted_noisy_location, delimiter=',', fmt='%d')

        return sorted_noisy_location

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            'Random samling: [Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()

        # normal samling
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train_normal):
            timer_data.hold()

            # data from normal sampling
            lr, hr = self.prepare([lr, hr])


            timer_model.tic()

            self.model.zero_grad()
            self.optimizer.zero_grad()

            sr = self.model(lr, idx_scale)
            loss = self.loss(sr, hr)

            loss.backward()
            self.optimizer.step()

            # rewind the weights, when the model is trained larger than 10 epoch
            # self.change_weight()
            if epoch > self.args.interval:
                # self.change_filter()
                self.change_weight()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train_normal.dataset),
                    self.loss.display_loss(batch),
                    loss.data.cpu().numpy().item(),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        save_path = self.args.save + '/inverse_model_training_un'+str(self.args.un_factor)+'/'
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), save_path + 'model_latest.pt' )

        self.loss.end_log(len(self.loader_train_normal))
        self.error_last = self.loss.log[-1, -1]

        # inverse sampling for training per 50 epochs
        if epoch % self.args.interval == 0:
            # train the model with inverse sampling data
            sorted_noisy_location = self.inverse_sampling(epoch)
            self.search_filter_index(sorted_noisy_location, epoch)

    def inverse_sampling(self, epoch):

        # self.scheduler.step()
        self.loss.step()
        # epoch = self.scheduler.last_epoch
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            'Inverse samling: [Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        # self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()

        # normal samling
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train_inverse):
            if batch > self.args.number:
                break
            timer_data.hold()

            # # data from normal sampling
            lr, hr = self.prepare([lr, hr])

            timer_model.tic()

            self.model.zero_grad()
            self.optimizer.zero_grad()

            sr = self.model(lr, idx_scale)
            l1_loss = self.loss(sr, hr)

            cl_loss = 0.0
            loss = l1_loss + cl_loss

            loss.backward()
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train_inverse.dataset),
                    self.loss.display_loss(batch),
                    l1_loss.data.cpu().numpy().item(),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        # save the current model trained with inverse sampling
        savepath = self.args.save + '/inverse_model_training_un'+str(self.args.un_factor)+'/'
        torch.save(self.model.state_dict(), os.path.join(savepath, 'Inverse_' + str(epoch) + '.pt'))
        # baseline_model_path = '../experiment/EDSR/EDSR_x4_WA_IG_filter_un100_0.01_Sy7/model/model_latest.pt'
        baseline_model_path = self.args.save + '/inverse_model_training_un'+str(self.args.un_factor)+'/model_latest.pt'
        target_model_path = self.args.save+'/inverse_model_training_un'+str(self.args.un_factor)+'/' + 'Inverse_' + str(
            epoch) + '.pt'
        sorted_noisy_location = self.extract_index(epoch, target_model_path, baseline_model_path)
        return sorted_noisy_location

    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        # rewind the weights
        # self.change_weight()
        # self.change_filter()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename, _) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare([lr, hr])
                    else:
                        lr = self.prepare([lr])[0]


                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )

                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

