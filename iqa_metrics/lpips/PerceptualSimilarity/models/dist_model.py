from __future__ import absolute_import

import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
from .base_model import BaseModel
from scipy.ndimage import zoom
from tqdm import tqdm


from . import networks_basic as networks
import iqa_metrics.lpips.PerceptualSimilarity.models as util


class DistModel(BaseModel):
    def name(self):
        return self.model_name

    def initialize(self, model='net-lin', net='alex', colorspace='Lab',
                   pnet_rand=False, pnet_tune=False, model_path=None,
                   use_gpu=True, printNet=False,
                   is_train=False, lr=.0001, beta1=0.5, version='0.1', gpu_ids=[0],
                   attention_type=None, attention_config=None):
        '''
        INPUTS
            model - ['net-lin'] for linearly calibrated network
                    ['net'] for off-the-shelf network
            net - ['alex','vgg']
            model_path - if None, will look in weights/[NET_NAME].pth
            colorspace - ['Lab','RGB'] colorspace to use for L2 and SSIM
            use_gpu - bool - whether or not to use a GPU
            printNet - bool - whether or not to print network architecture out

            is_train - bool - [True] for training mode
            lr - float - initial learning rate
            beta1 - float - initial momentum term for adam
            version - 0.1 for latest, 0.0 was original (with a bug)
            gpu_ids - int array - [0] by default, gpus to use
        '''
        BaseModel.initialize(self, use_gpu=use_gpu, gpu_ids=gpu_ids)

        print("Initializing LPIPS model...")

        self.model = model
        self.pnet_type = net
        self.is_train = is_train
        self.gpu_ids = gpu_ids
        self.model_name = '%s [%s]' % (model, net)

        # if using pre-trained original LPIPS, disable attention
        if not is_train:
            attention_type = None

        if self.model == "a2":
            print("PNet type: PNetA2.")

            self.net = networks.PNetA2(pnet_rand=pnet_rand, pnet_tune=pnet_tune)

            if model_path is None:
                import inspect
                model_path = os.path.abspath(
                    os.path.join(inspect.getfile(self.initialize), '..', 'weights/a2/best.pth'))
                self.model_path = model_path

        elif self.model == 'net-lin':  # pretrained net + linear layer
            print("PNet type: PNetLin with", net)

            self.net = networks.PNetLin(pnet_type=net, pnet_rand=pnet_rand, pnet_tune=pnet_tune,
                                        use_dropout=True, version=version,
                                        attention_type=attention_type,
                                        attention_config=attention_config)

            if model_path is None:
                import inspect
                if attention_type == "SAGAN":
                    model_path = os.path.abspath(
                        os.path.join(inspect.getfile(self.initialize), '..', 'weights/sagan/best.pth'))
                elif self.pnet_type == "resnet50":
                    model_path = os.path.abspath(
                        os.path.join(inspect.getfile(self.initialize), '..', 'weights/resnet50/best.pth'))
                else:
                    model_path = os.path.abspath(
                        os.path.join(inspect.getfile(self.initialize), '..', 'weights/v%s/%s.pth' % (version, net)))

                    kw = {}
                    if not use_gpu:
                        kw['map_location'] = 'cpu'

                    if not is_train:
                        print('Loading LPIPS model from: %s' % model_path)
                        self.net.load_state_dict(torch.load(model_path, **kw), strict=False)

                self.model_path = model_path

        elif self.model == 'net':  # pretrained network
            self.net = networks.PNetLin(pnet_rand=pnet_rand, pnet_type=net,
                                        attention_type=attention_type,
                                        attention_config=attention_config)
        else:
            raise ValueError("Model [%s] not recognized." % self.model)

        self.parameters = list(self.net.parameters())

        # print("trainable params")
        # print(self.parameters)

        if self.is_train:  # training mode
            # extra network on top to go from distances (d0,d1) => predicted human judgment (h*)
            self.rankLoss = networks.BCERankingLoss()
            self.parameters += list(self.rankLoss.net.parameters())
            self.lr = lr
            self.old_lr = lr
            self.optimizer_net = torch.optim.Adam(self.parameters, lr=lr, betas=(beta1, 0.999))
            self.train()  # original didnt have this
        else:  # test mode
            self.eval()  # original -> self.net.eval()

        if use_gpu:
            self.net.to(gpu_ids[0])
            self.net = torch.nn.DataParallel(self.net, device_ids=gpu_ids)
            if self.is_train:
                self.rankLoss = self.rankLoss.to(device=gpu_ids[0])  # small net, just put this on GPU0

        if printNet:
            print('---------- Networks initialized -------------')
            networks.print_network(self.net)
            print('-----------------------------------------------')

    # def train(self):
    #     self.net.train()
    #
    # def eval(self):
    #     self.net.eval()

    def forward(self, in0, in1, retPerLayer=False):
        ''' Function computes the distance between image patches in0 and in1
        INPUTS
            in0, in1 - torch.Tensor object of shape Nx3xXxY - image patch scaled to [-1,1]
        OUTPUT
            computed distances between in0 and in1
        '''

        return self.net.forward(in0, in1, retPerLayer=retPerLayer)

    # ***** TRAINING FUNCTIONS *****
    def optimize_parameters(self):
        self.optimizer_net.zero_grad()
        self.forward_train()
        self.backward_train()
        self.optimizer_net.step()
        self.clamp_weights()

    def clamp_weights(self):
        for module in self.net.modules():
            if (hasattr(module, 'weight') and hasattr(module, 'kernel_size') and module.kernel_size == (1, 1)):
                module.weight.data = torch.clamp(module.weight.data, min=0)

    def set_input(self, data):
        self.input_ref = data['ref']
        self.input_p0 = data['p0']
        self.input_p1 = data['p1']
        self.input_judge = data['judge']

        if self.use_gpu:
            self.input_ref = self.input_ref.to(device=self.gpu_ids[0])
            self.input_p0 = self.input_p0.to(device=self.gpu_ids[0])
            self.input_p1 = self.input_p1.to(device=self.gpu_ids[0])
            self.input_judge = self.input_judge.to(device=self.gpu_ids[0])

        self.var_ref = Variable(self.input_ref, requires_grad=True)
        self.var_p0 = Variable(self.input_p0, requires_grad=True)
        self.var_p1 = Variable(self.input_p1, requires_grad=True)

    def forward_train(self):  # run forward pass
        self.d0 = self.forward(self.var_ref, self.var_p0)
        self.d1 = self.forward(self.var_ref, self.var_p1)

        self.acc_r = self.compute_accuracy(self.d0, self.d1, self.input_judge)

        self.var_judge = Variable(1. * self.input_judge).view(self.d0.size())

        self.loss_total = self.rankLoss.forward(self.d0, self.d1, self.var_judge * 2. - 1.)

        # print("self.loss_total", self.loss_total)

        return self.loss_total

    def backward_train(self):
        torch.mean(self.loss_total).backward()

    def compute_accuracy(self, d0, d1, judge):
        ''' d0, d1 are Variables, judge is a Tensor '''
        d1_lt_d0 = (d1 < d0).cpu().data.numpy().flatten()
        judge_per = judge.cpu().numpy().flatten()
        return d1_lt_d0 * judge_per + (1 - d1_lt_d0) * (1 - judge_per)

    def get_current_errors(self):
        retDict = OrderedDict([('loss_total', self.loss_total.data.cpu().numpy()),
                               ('acc_r', self.acc_r)])

        for key in retDict.keys():
            retDict[key] = np.mean(retDict[key])

        return retDict

    def get_current_visuals(self):
        zoom_factor = 256 / self.var_ref.data.size()[2]

        ref_img = util.tensor2im(self.var_ref.data)
        p0_img = util.tensor2im(self.var_p0.data)
        p1_img = util.tensor2im(self.var_p1.data)

        ref_img_vis = zoom(ref_img, [zoom_factor, zoom_factor, 1], order=0)
        p0_img_vis = zoom(p0_img, [zoom_factor, zoom_factor, 1], order=0)
        p1_img_vis = zoom(p1_img, [zoom_factor, zoom_factor, 1], order=0)

        return OrderedDict([('ref', ref_img_vis),
                            ('p0', p0_img_vis),
                            ('p1', p1_img_vis)])

    def restore_checkpoint(self, filename, device):
        """Load model from a checkpoint"""
        print("Loading model parameters from '{}'".format(filename))
        with open(filename, "rb") as f:
            checkpoint_data = torch.load(f, map_location=device)

        try:
            self.load_state_dict(checkpoint_data["model_state_dict"])
        except RuntimeError as e:
            print("Missing state_dict keys in checkpoint", "red", e)
            print("Retry import with current model values for missing keys.")
            state = self.state_dict()
            state.update(checkpoint_data["model_state_dict"])
            self.load_state_dict(state)

    def save(self, path, label):
        if self.use_gpu:
            self.save_network(self.net.module, path, '', label)
        else:
            self.save_network(self.net, path, '', label)
        self.save_network(self.rankLoss.net, path, 'rank', label)

    def lr_decay_linear(self, nepoch_decay):
        lrd = self.lr / nepoch_decay
        lr = self.old_lr - lrd

        for param_group in self.optimizer_net.param_groups:
            param_group['lr'] = lr

        print('update lr [%s] decay: %f -> %f' % (type, self.old_lr, lr))
        self.old_lr = lr

    def lr_decay(self, decay):
        lr = decay * self.old_lr

        for param_group in self.optimizer_net.param_groups:
            param_group['lr'] = lr

        print('update lr [%s] decay: %f -> %f' % (type, self.old_lr, lr))
        self.old_lr = lr


def score_2afc_dataset(data_loader, func, name=''):
    ''' Function computes Two Alternative Forced Choice (2AFC) score using
        distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a TwoAFCDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return numpy array of length N
    OUTPUTS
        [0] - 2AFC score in [0,1], fraction of time func agrees with human evaluators
        [1] - dictionary with following elements
            d0s,d1s - N arrays containing distances between reference patch to perturbed patches
            gts - N array in [0,1], preferred patch selected by human evaluators
                (closer to "0" for left patch p0, "1" for right patch p1,
                "0.6" means 60pct people preferred right patch, 40pct preferred left)
            scores - N array in [0,1], corresponding to what percentage function agreed with humans
    CONSTS
        N - number of test triplets in data_loader
    '''

    d0s = []
    d1s = []
    gts = []

    for data in tqdm(data_loader.load_data(), desc=name):
        d0s += func(data['ref'], data['p0']).data.cpu().numpy().flatten().tolist()
        d1s += func(data['ref'], data['p1']).data.cpu().numpy().flatten().tolist()
        gts += data['judge'].cpu().numpy().flatten().tolist()

    d0s = np.array(d0s)
    d1s = np.array(d1s)
    gts = np.array(gts)
    scores = (d0s < d1s) * (1. - gts) + (d1s < d0s) * gts + (d1s == d0s) * .5

    return (np.mean(scores), dict(d0s=d0s, d1s=d1s, gts=gts, scores=scores))


def score_jnd_dataset(data_loader, func, name=''):
    ''' Function computes JND score using distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a JNDDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return pytorch array of length N
    OUTPUTS
        [0] - JND score in [0,1], mAP score (area under precision-recall curve)
        [1] - dictionary with following elements
            ds - N array containing distances between two patches shown to human evaluator
            sames - N array containing fraction of people who thought the two patches were identical
    CONSTS
        N - number of test triplets in data_loader
    '''

    ds = []
    gts = []

    for data in tqdm(data_loader.load_data(), desc=name):
        ds += func(data['p0'], data['p1']).data.cpu().numpy().tolist()
        gts += data['same'].cpu().numpy().flatten().tolist()

    sames = np.array(gts)
    ds = np.array(ds)

    sorted_inds = np.argsort(ds)
    ds_sorted = ds[sorted_inds]
    sames_sorted = sames[sorted_inds]

    TPs = np.cumsum(sames_sorted)
    FPs = np.cumsum(1 - sames_sorted)
    FNs = np.sum(sames_sorted) - TPs

    precs = TPs / (TPs + FPs)
    recs = TPs / (TPs + FNs)
    score = util.voc_ap(recs, precs)

    return (score, dict(ds=ds, sames=sames))
