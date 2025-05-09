# coding=utf-8
import os
import time
import json
import logging
import math
import argparse
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.multiprocessing as mp
from torch import nn
from copy import deepcopy
import model
from util.timer import Timer
from util import args_processing as ap
from util import consts
from util import env
from util import new_metrics
from loader import multi_metric_meta_sequence_dataloader as meta_sequence_dataloader
import numpy as np
from util import utils
# from transformers import AutoTokenizer
from model.cocorrrec.ttt import TTTConfig
from model.cocorrrec.cocorrrec_model import CoCorrRec
utils.setup_seed(0)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=str, help="Kernels configuration for CNN")
    parser.add_argument("--bucket", type=str, default=None, help="Bucket name for external storage")
    parser.add_argument("--dataset", type=str, default="alipay", help="Bucket name for external storage")
    parser.add_argument("--data",type=str,default='amazon_beauty',help="dataset name such as amazon_beauty")
    parser.add_argument("--max_steps", type=int, help="Number of iterations before stopping")
    parser.add_argument("--snapshot", type=int, help="Number of iterations to dump model")
    parser.add_argument("--checkpoint_dir", type=str, help="Path of the checkpoint path")
    parser.add_argument("--learning_rate", type=str, default=0.001)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--max_epoch", type=int,  default=10, help="Max epoch")
    parser.add_argument("--num_loading_workers", type=int, default=4, help="Number of threads for loading")
    parser.add_argument("--model", type=str, help="model type")
    parser.add_argument("--init_checkpoint", type=str, default="", help="Path of the checkpoint path")
    parser.add_argument("--init_step", type=int, default=0, help="Path of the checkpoint path")

    parser.add_argument("--max_gradient_norm", type=float, default=0.)

    parser.add_argument("--arch_config_path", type=str, default=None, help="Path of model configs")
    parser.add_argument("--arch_config", type=str, default=None, help="base64-encoded model configs")

    parser.add_argument("--mini_batch_size", type=int, default=1, help="base learning rate for TTT learner")
    parser.add_argument("--mode", type=str, default='base', help="")
    parser.add_argument("--initializer_range", type=float, default=0.02, help="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.")
    return parser.parse_known_args()[0]


def predict(predict_dataset, model_obj, device, args, train_epoch, train_step, writer=None):

    model_obj.eval()
    model_obj.to(device)

    timer = Timer()
    log_every = 200

    pred_list = []
    y_list = []
    buffer = []
    user_id_list = []
    for step, batch_data in enumerate(predict_dataset, 1):
        logits = model_obj({
            key: value.to(device)
            for key, value in batch_data.items()
            if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
        })
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        y = batch_data[consts.FIELD_LABEL].view(-1, 1)
        overall_auc, _, _, _ = new_metrics.calculate_overall_auc(prob, y)

        user_id_list.extend(np.array(batch_data[consts.FIELD_USER_ID].view(-1, 1)))
        pred_list.extend(prob)
        y_list.extend(np.array(y))

        buffer.extend(
            [int(user_id), float(score), float(label)]
            for user_id, score, label
            in zip(
                batch_data[consts.FIELD_USER_ID],
                prob,
                batch_data[consts.FIELD_LABEL]
            )
        )

        if step % log_every == 0:
            logger.info(
                "train_epoch={}, step={}, overall_auc={:5f}, speed={:2f} steps/s".format(
                    train_epoch, step, overall_auc, log_every / timer.tick(False)
                )
            )

    overall_auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred_list), np.array(y_list))
    user_auc = new_metrics.calculate_user_auc(buffer)
    overall_logloss = new_metrics.calculate_overall_logloss(np.array(pred_list), np.array(y_list))
    user_ndcg5, user_hr5 = new_metrics.calculate_user_ndcg_hr(5, buffer)
    user_ndcg10, user_hr10 = new_metrics.calculate_user_ndcg_hr(10, buffer)
    user_ndcg20, user_hr20 = new_metrics.calculate_user_ndcg_hr(20, buffer)

    print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
          "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
          format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                 user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20))
    with open(os.path.join(args.checkpoint_dir, "log_ood.txt"), "a") as writer:
        print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
              "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
              format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                     user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20), file=writer)

    return overall_auc, user_auc, overall_logloss, user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20


def train(train_dataset, model_obj, device, args, pred_dataloader):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model_obj.parameters(),
        lr=float(args.learning_rate)
    )
    model_obj.train()
    model_obj.to(device)

    print(model_obj)
    if args.mode == 'base':
        for layer in model_obj.ttt.layers:
            layer.seq_modeling_block.correction = False
    else:
        for layer in model_obj.ttt.layers:
            layer.seq_modeling_block.correction = True
    logger.info("Start training...")
    timer = Timer()
    log_every = 200
    max_step = 0
    best_auc = 0
    best_auc_ckpt_path = os.path.join(args.checkpoint_dir, "best_auc" + ".pkl")
    # total_params = sum(p.numel() for p in model_obj.parameters())
    # with open(os.path.join(args.checkpoint_dir, "log_ood.txt"), "a") as writer:
    #     print(f"{args.model} total paras: {total_params}",file=writer)
    #     exit(0)
    for epoch in range(1, args.max_epoch + 1):
        for step, batch_data in enumerate(train_dataset):
            logits = model_obj({
                key: value.to(device)
                for key, value in batch_data.items()
                if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}})

            loss = criterion(logits, batch_data[consts.FIELD_LABEL].view(-1, 1).to(device))
            pred, y = torch.sigmoid(logits), batch_data[consts.FIELD_LABEL].view(-1, 1)
            auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred.detach().cpu()), np.array(y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % log_every == 0:
                logger.info(
                    "epoch={}, step={}, loss={:5f}, auc={:5f}, speed={:2f} steps/s".format(
                        epoch, step, float(loss.item()), auc, log_every / timer.tick(False)
                    )
                )
            max_step = step

        pred_overall_auc, pred_user_auc, pred_overall_logloss, pred_user_ndcg5, pred_user_hr5, \
        pred_user_ndcg10, pred_user_hr10, pred_user_ndcg20, pred_user_hr20 = predict(
            predict_dataset=pred_dataloader,
            model_obj=model_obj,
            device=device,
            args=args,
            train_epoch=epoch,
            train_step=epoch * max_step,
        )
        model_obj.train()
        if pred_overall_auc > best_auc:
            logger.info("dump checkpoint for epoch {}".format(epoch))
            best_auc = pred_overall_auc
            torch.save(model_obj, best_auc_ckpt_path)




def prepare_dataloder(args,train_all_file,test_all_file):
    worker_id = worker_count = 8

    train_all_dataloader = meta_sequence_dataloader.MetaSequenceDataLoader(
        table_name=train_all_file,
        slice_id=0,
        slice_count=args.num_loading_workers,
        is_train=True
    )
    train_all_dataloader = torch.utils.data.DataLoader(
        train_all_dataloader,
        batch_size=args.batch_size,
        num_workers=args.num_loading_workers,
        pin_memory=True,
        collate_fn=train_all_dataloader.batchify,
        drop_last=False
    )

    # Setup up data loader
    pred_all_dataloader = meta_sequence_dataloader.MetaSequenceDataLoader(
        table_name=test_all_file,
        slice_id=args.num_loading_workers * worker_id,
        slice_count=args.num_loading_workers * worker_count,
        is_train=False
    )
    pred_all_dataloader = torch.utils.data.DataLoader(
        pred_all_dataloader,
        batch_size=args.batch_size,
        num_workers=args.num_loading_workers,
        pin_memory=True,
        collate_fn=pred_all_dataloader.batchify,
        drop_last=False
    )
    return train_all_dataloader,pred_all_dataloader




def main_worker(_):
    args = parse_args()
    ap.print_arguments(args)
    torch.cuda.set_device('cuda:0')
    
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    with open(args.arch_config,'r') as f:
        data_config = json.load(f)

    data_config['id_dimension'] = 64
    initializer_range_dict = {'yelp':0.03,'amazon_beauty':0.02,'amazon_electronic':0.03}
    initializer_range = initializer_range_dict[args.data]
    mini_batch_size = 10
    configuration = TTTConfig(initializer_range = initializer_range,ttt_layer_type = 'linear',vocab_size = data_config['id_vocab'],hidden_size = data_config['id_dimension'],mini_batch_size=mini_batch_size, pad_token_id=0,num_attention_heads=data_config['nhead'],num_hidden_layers = 1,mode = args.mode)

    # Initializing a model from the ttt-1b style configuration
    model_obj = CoCorrRec(data_config,configuration)
    model_obj.eval()
    device = env.get_device()

    train_all_file,test_all_file = args.dataset.split(',')
    
    args.num_loading_workers = 1
    train_all_dataloader,pred_all_dataloader = prepare_dataloder(args,train_all_file,test_all_file)
    # Setup training
    train(
        train_dataset=train_all_dataloader,
        model_obj=model_obj,
        device=device,
        args=args,
        pred_dataloader=pred_all_dataloader
    )



if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main_worker(1)

