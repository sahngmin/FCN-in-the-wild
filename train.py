#!/usr/bin/env python

import argparse
import os
import os.path as osp
import torch
import datetime
from torch.utils import data
from deeplab_multi import Deeplab_multi
from dataset.gta5_dataset import GTA5DataSet
from dataset.synthia_dataset import SYNTHIADataSet
from dataset.cityscapes_dataset import CityScapesDataSet
from dataset.idd_dataset import IDDDataSet
from data.GTA5 import GTA5
from FCN.model import FCN
from FCN.trainer import Trainer
from data.data_utils  import get_label_classes
from FCN.vgg import VGG16
import pdb 
from options_train import TrainOptions
import torch.optim as optim



here = osp.dirname(osp.abspath(__file__))

def get_parameters(model, bias=False):
	import torch.nn as nn
	modules_skipped = (
		FCN,
		nn.ReLU,
		nn.MaxPool2d,
		nn.Dropout2d,
		nn.Sequential,
		nn.Upsample
	)
	#pdb.set_trace()
	for m in model.modules():
		if isinstance(m, nn.Conv2d):
			if bias:
				yield m.bias
			else:
				yield m.weight
		elif isinstance(m, nn.ConvTranspose2d):
			# weight is frozen because it is just a bilinear upsampling
			if bias:
				assert m.bias is None
		elif isinstance(m, modules_skipped):
			continue
		else:
			raise ValueError('Unexpected module: %s' % str(m))

def main():
	args = TrainOptions().parse()
	resume = args.resume
	out = os.getcwd()
	cuda = torch.cuda.is_available()
	
	if cuda:
		torch.cuda.manual_seed(1123)
	else:
		torch.manual_seed(1123)
	
	# load dataset
	root = osp.join(here, 'data')
	kwargs = {'num_workers': 2, 'pin_memory': True} if cuda else {}


	# train_loader = torch.utils.data.DataLoader(
	# 	GTA5(root, split='train', transform=True),
	# 	batch_size=1, shuffle=True, **kwargs)

	# val_loader = torch.utils.data.DataLoader(
	# 	GTA5(root, split='val', transform=True),
	# 	batch_size=1, shuffle=False, **kwargs)

	input_size = (1024, 512)

	train_loader = data.DataLoader(
		GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
					crop_size=input_size, ignore_label=args.ignore_label,
					set=args.set, num_classes=args.num_classes),
		batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)


	# model
	model = Deeplab_multi(args=args)

	start_epoch = 0
	start_iteration = 0

	check_point = None


	if cuda:
		model = model.cuda()

	# optimizer
	optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
	# if resume:
	# 	optim.load_state_dict(check_point['optim_state_dict'])

	trainer = Trainer(
		cuda=cuda,
		model=model,
		optimizer=optimizer,
		train_loader=train_loader,
		out=out,
		max_iter=args.num_steps_stop,
		interval_validate=args.save_pred_every,
		args=args
	)
	trainer.epoch = start_epoch
	trainer.iteration = start_iteration
	trainer.train()

if __name__ == '__main__':
	from __init_path__ import add_full_path
	add_full_path()
	from utils.util import label_accuracy_score
	main()
	