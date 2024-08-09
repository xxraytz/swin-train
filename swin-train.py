import torch
import json
import os
from datasets import load_dataset
import numpy as np
from logger import create_logger
import random
import datetime

try:
    from torch._six import inf
except:
    from torch import inf

from timm.data import create_transform, Mixup
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomCrop,
    Resize,
    ToTensor,
)

try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp

from torch.utils.data import DataLoader

from torch import nn
from transformers import SwinConfig, SwinForImageClassification
from tqdm import tqdm
from torch import optim
import time
import requests
from torch.cuda.amp import GradScaler
from torch.profiler import record_function, ProfilerActivity, profile
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.utils import accuracy, AverageMeter
from functools import partial
from timm.loss import LabelSmoothingCrossEntropy
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
from config import get_config
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from utils import (load_checkpoint,
                   save_checkpoint,
                   NativeScalerWithGradNormCount, auto_resume_helper, reduce_tensor)

# Linear layers
try:
    from bitsandbytes.nn.triton_based_modules import SwitchBackLinear
    from bitsandbytes.optim import AdamW8bit, Adam8bit
except:
    print('No bitsandbytes lib are installed. You could fall into troubles with Adam8bit, AdamW8bit, SwitchbackLinear')

import sys


sys.path.append("/home/dev/Jetfire-INT8Training/JetfireGEMMKernel/BlocknviQuantize/")
try:
    from EQBlockLinear import EQBlockLinear
except:
    print('No bitsandbytes lib are installed. You could get stacked with JetFire linear layer')


PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=False, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    # for pytorch >= 2.0, use `os.environ['LOCAL_RANK']` instead
    # (see https://pytorch.org/docs/stable/distributed.html#launch-utility)
    if PYTORCH_MAJOR_VERSION == 1:
        parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    parser.add_argument('--token', type=str, 
                        help='Put your huggingface token here to download dataset.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def send_telegram_msg(text):
    token = "5624864341:AAECudDmaSYMfF0qKt2bIPT-YCrLhPXTmaI"
    chat_id = "-1001685956068"
    url_req = "https://api.telegram.org/bot" + token + "/sendMessage" + "?chat_id=" + chat_id + "&text=" + text
    requests.get(url_req)


def get_labels(a_dataset):
    labels = a_dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    # id2label[2]
    return labels, label2id, id2label


def get_ds(dataset_name, data_path, num_proc=4):
    dataset = load_dataset(dataset_name,
#                            split='train[:50000]',
                           num_proc=num_proc,
                           cache_dir=data_path,
                           token=os.environ['TOKEN']
                           )
    print(dataset)
#     dataset = dataset.train_test_split(0.2, )
#     test_name = 'test'
    train_name, test_name = list(dataset.keys())[:2]

    train_len = len(dataset[train_name])
    test_len = len(dataset[test_name])

    labels, label2id, id2label = get_labels(dataset)
    # split up training into training + validation
    train_ds = dataset[train_name]
    test_ds = dataset[test_name]
    # val_ds = dataset['validation']

    return train_ds, test_ds, labels, label2id, id2label, train_len, test_len


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                       interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(ToTensor())
    t.append(Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return Compose(t)


def get_loaders(train_ds, test_ds, config):

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    test_dataloader = DataLoader(
        test_ds,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return train_dataloader, test_dataloader, mixup_fn


def preprocess(example_batch, transform_f):
    example_batch["pixel_values"] = [
        transform_f(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def replace_all_linear_layers(module, custom_linear_layer, index=0):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if (child.in_features % 32 == 0 
                and child.out_features % 32 == 0 
                and child.out_features > (32 * 3)
                and (child.in_features != 192 and child.out_features != 192)
                ):
                # print(index)
                index += 1
                if index in []:
                    logger.info(f'{child}-no')
                    new_linear = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None)
                else:
                    logger.info(f'{child}-yes')
                    new_linear = custom_linear_layer(child.in_features, child.out_features, bias=child.bias is not None)
            else:
                logger.info(f'{child}-no')
                new_linear = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None)
            # new_linear.weight.data = child.weight.data.half()
            # if new_linear.bias is not None:
            # new_linear.bias.data = child.bias.data.half()
            setattr(module, name, new_linear)
        else:
            index = replace_all_linear_layers(child, custom_linear_layer, index=index)
    return index

def get_model(config, device, num_labels, label2id, id2label, linear='simple'):
    # config.window_size  = 8
    
    
    swin_config = SwinConfig(
        img_size=config.DATA.IMG_SIZE,
        patch_size=config.MODEL.SWIN.PATCH_SIZE,
        # in_chans=config.MODEL.SWIN.IN_CHANS,
        # num_classes=config.MODEL.NUM_CLASSES,
        embed_dim=config.MODEL.SWIN.EMBED_DIM,
        depths=config.MODEL.SWIN.DEPTHS,
        num_heads=config.MODEL.SWIN.NUM_HEADS,
        window_size=config.MODEL.SWIN.WINDOW_SIZE,
        # mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
        # qkv_bias=config.MODEL.SWIN.QKV_BIAS,
        # qk_scale=config.MODEL.SWIN.QK_SCALE,
        # drop_rate=config.MODEL.DROP_RATE,
        # drop_path_rate=config.MODEL.DROP_PATH_RATE,
        # ape=config.MODEL.SWIN.APE,
        # norm_layer=layernorm,
        # patch_norm=config.MODEL.SWIN.PATCH_NORM,
        # use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        # fused_window_process=config.FUSED_WINDOW_PROCESS,
        num_labels=num_labels, label2id=label2id, id2label=id2label
        )

    
    m = SwinForImageClassification(swin_config)
    if linear == 'switchback':
        replace_all_linear_layers(m, SwitchBackLinear)
    elif linear == 'jetfire':
        # Here we put our jetfire code.

        replace_all_linear_layers(m, EQBlockLinear)
        ...
    print(m)
    m = m.to(device)
    return m


def get_criterion():
    return LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)


def get_optimizer(config, model):
    parameters = set_weight_decay(model, {}, {})
    if os.environ["OPTIMIZER"] == 'AdamW':
        optim_f = optim.AdamW
    elif os.environ["OPTIMIZER"] == 'AdamW8bit':
        optim_f = AdamW8bit
        
    logger.info(f'{optim_f} was choosed.')
    optimizer = optim_f(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def get_lr_scheduler(config, optimizer, n_epochs, n_iter_per_epoch, train_warmup_epochs):
    print(n_epochs, n_iter_per_epoch, train_warmup_epochs)
    num_steps = int(n_epochs * n_iter_per_epoch)
    warmup_steps = int(train_warmup_epochs * n_iter_per_epoch)
    print(num_steps, warmup_steps)
    cosine_lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=20,#(num_steps - warmup_steps),
        # t_mul=1.,
        lr_min=config.TRAIN.MIN_LR,
        warmup_lr_init=config.TRAIN.WARMUP_LR,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
        warmup_prefix=config.TRAIN.LR_SCHEDULER.WARMUP_PREFIX,
    )
    return cosine_lr_scheduler


def train_one_epoch(config, model, criterion, train_data_loader, optimizer, epoch,
                    mixup_fn, lr_scheduler, loss_scaler,
                    fp16=False):
    optimizer.zero_grad()

    num_steps = len(train_data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    
    torch.cuda.synchronize()
    start = time.time()
    end = time.time()
    # from itertools import islice
    for idx, data in enumerate(train_data_loader):
        inputs = data['pixel_values'].cuda(non_blocking=True)
        labels = data['labels'].cuda(non_blocking=True)

        if mixup_fn is not None:
            inputs, labels = mixup_fn(inputs, labels)
            labels = torch.argmax(labels, 1)

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            outputs = model(inputs)

        loss = criterion(outputs.logits, labels.long())
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=1,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)

        optimizer.zero_grad()
        lr_scheduler.step_update(epoch * num_steps + idx)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()
        loss_meter.update(loss.item(), labels.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    

@torch.no_grad()
def validate(config, data_loader, model, fp16=False):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()

    for idx, data in enumerate(data_loader):
        inputs = data['pixel_values'].cuda()
        labels = data['labels'].cuda()
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(inputs)

        # measure accuracy and record loss
        loss = criterion(output.logits, labels)
        acc1, acc5 = accuracy(output.logits, labels, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), labels.size(0))
        acc1_meter.update(acc1.item(), labels.size(0))
        acc5_meter.update(acc5.item(), labels.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


def main(config):
    print("Number of available GPUs:", torch.cuda.device_count())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, test_ds, labels, label2id, id2label, train_len, test_len = get_ds("imagenet-1k", config.DATA.DATA_PATH, 
                                                                                num_proc=config.DATA.NUM_WORKERS)

    train_transforms = build_transform(True, config)
    val_transforms = build_transform(False, config)

    train_ds.set_transform(partial(preprocess, transform_f=train_transforms))
    test_ds.set_transform(partial(preprocess, transform_f=val_transforms))

    train_dataloader, test_dataloader, mixup = get_loaders(train_ds, test_ds, config)

    
    swin_model = get_model(config, device, len(labels), label2id, id2label, linear=os.environ['LINEAR_TYPE'])
    model_without_ddp = swin_model
    loss_scaler = NativeScalerWithGradNormCount()

    swin_optimizer = get_optimizer(
        config,
        swin_model,
    )

    swin_lr_scheduler = get_lr_scheduler(
        config,
        swin_optimizer,
        n_epochs=config.TRAIN.EPOCHS,
        n_iter_per_epoch=len(train_dataloader),
        train_warmup_epochs=config.TRAIN.WARMUP_EPOCHS,
    )

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, swin_optimizer, swin_lr_scheduler, loss_scaler,
                                       logger)
        acc1, acc5, loss = validate(config, test_dataloader, swin_model)
        logger.info(f"Accuracy of the network on the {len(test_ds)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):

        train_one_epoch(
            config, swin_model, get_criterion(), train_dataloader,
            swin_optimizer, epoch, mixup, swin_lr_scheduler,
            loss_scaler
        )
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, swin_optimizer, swin_lr_scheduler,
                            loss_scaler,
                            logger)
        # for data in test_dataloader:
        #     print(data['labels'][0])
        #     break

        acc1, acc5, loss = validate(config, test_dataloader, swin_model)
        logger.info(f"Accuracy of the network on the {len(test_ds)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    print(config.TRAIN.BASE_LR, config.DATA.BATCH_SIZE, dist.get_world_size())
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)