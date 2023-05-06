import argparse
import logging
import os
from pathlib import Path
import yaml

from train import train
from utils.general import (
    increment_path,
    check_file,
    set_logging)
from utils.torch_utils import select_device
from otm.utils import folder_cleaner_coco, file_mover_coco, file_sampler_coco, file_copier_coco
from otm.server import TCPServer

logger = logging.getLogger(__name__)


def online_training(hyp, opt, device):
    """

    :param hyp: hyperparameters yaml path
    :param opt: args NameSpace for parsed cli arguments
    :param device: cuda device id
    :return:
    """

    dataset_dir = opt.memory_path
    incoming_dir = opt.stream_path
    temp_dir = opt.temp_path  # used for training
    replay_file_nb = opt.replay_sample_nb  # num samples to select from memory
    threshold = opt.threshold

    files_list = [f for f in os.listdir(incoming_dir + "/labels")]
    # Get the count of files in the directory
    files_count = len(files_list)
    print(files_count)

    if files_count > threshold:
        file_copier_coco(incoming_dir, temp_dir)
        file_sampler_coco(dataset_dir, temp_dir, replay_file_nb)
        final_model_path, _ = train(hyp, opt, device, tb_writer=None)

        # Server codes below.
        server2 = TCPServer(host='169.254.153.152', port=65432, folder_path=final_model_path)
        server2.start()
        server2.serve()
        server2.close()

        file_mover_coco(incoming_dir, dataset_dir)
        folder_cleaner_coco(temp_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--memory_path', type=str,
        default='./otm/custom_dataset/train')
    parser.add_argument(
        '--stream_path', type=str,
        default='./otm/incoming_folder')
    parser.add_argument(
        '--replay_sample_nb', type=int,
        default=100, help="give number of samples to use for replay")
    parser.add_argument(
        '--temp_path', type=str,
        default='./otm/temp_folder')
    parser.add_argument(
        '--device', default='0',
        help='cuda device id')
    parser.add_argument(
        '--epochs', type=int,
        default=300)
    parser.add_argument(
        '--weights', type=str,
        default='./otm/base_weights/yolo7.pt',
        help='initial weights path')
    parser.add_argument(
        '--threshold', type=int, default=10,
        help='minimum file number to start training')
    parser.add_argument(
        '--batch-size', type=int,
        default=16, help='total batch size for all GPUs')
    parser.add_argument(
        '--total_batch_size', type=int,
        default=32, help='Total batch size')
    parser.add_argument(
        '--img-size', nargs='+', type=int,
        default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--hyp', type=str,
                        default='data/hyp.scratch.p5.yaml',
                        help='hyperparameters path')

    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0],
                        help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')

    opt = parser.parse_args()

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)

    # check files
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.name = 'evolve' if opt.evolve else opt.name
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    logger.info(opt)
    device = select_device(opt.device, batch_size=opt.batch_size)
    online_training(hyp, opt, device)
