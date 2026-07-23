import os
import torch


class Options():
    """docstring for Options"""

    def __init__(self):
        pass

    def init(self, parser):
        # global settings
        parser.add_argument('--batch_size', type=int, default=3, help='batch size')  # 24 for vallina omnisr, 15 for omnisrfreq, 10 for large scale omnisrfreq
        parser.add_argument('--nepoch', type=int, default=5000, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=4, help='train_dataloader workers') 
        parser.add_argument('--eval_workers', type=int, default=2, help='eval_dataloader workers')
        parser.add_argument('--dataset', type=str, default='render_data')
        parser.add_argument('--pretrain_weights', type=str, default=None,help='path of pretrained_weights')
        parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
        parser.add_argument('--arch', type=str, default='DenseSR', help='archtechture')
        parser.add_argument('--mode', type=str, default='shadow', help='image restoration mode')

        # args for saving
        # parser.add_argument('--save_dir', type=str, default='./NTIRE_log', help='save dir')
        parser.add_argument('--save_dir', type=str, default='DenseSR', help='save dir')
        parser.add_argument('--save_images', action='store_true', default=False)
        parser.add_argument('--env', type=str, default='Debug', help='env')
        parser.add_argument('--checkpoint', type=int, default=3, help='checkpoint')

        # args for Uformer
        parser.add_argument('--norm_layer', type=str, default='nn.LayerNorm', help='normalize layer in transformer')
        parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
        parser.add_argument('--win_size', type=int, default=16, help='window size of self-attention')
        parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
        parser.add_argument('--token_mlp', type=str, default='leff', help='ffn/leff token mlp')
        parser.add_argument('--att_se', action='store_true', default=False, help='se after sa')



        # args for training
        parser.add_argument('--debug', action='store_true', default=False, help='debug model')
        parser.add_argument('--eval_now', type=int, default=3, help='After how many epochs to evaluate')
        parser.add_argument('--test_every', type=int, default=3, help='After how many epochs to run region test metrics')
        parser.add_argument('--test_limit', type=int, default=0, help='limit number of test images for smoke checks; 0 means full test set')
        parser.add_argument('--metric_eval_size', type=int, default=256, help='resize size for DESOBA region metrics')
        parser.add_argument('--val_batch_size', type=int, default=1, help='validation batch size')
        parser.add_argument('--max_train_steps', type=int, default=0, help='max train steps per epoch for probing; 0 means full epoch')
        parser.add_argument('--shadow_loss_weight', type=float, default=0.0, help='extra Charbonnier loss weight on shadow-mask pixels')
        parser.add_argument('--ssim_loss_weight', type=float, default=0.0, help='optional full-image SSIM loss weight during training')
        parser.add_argument('--train_ps', type=int, default=512, help='patch size of training sample')
        parser.add_argument('--resume', action='store_true', default=False)
        parser.add_argument('--dino_dim', type=int, default=1024, help='dim of dino features')
        parser.add_argument('--dino_version', type=str, default='dinov3', help='dino version: dinov2 or dinov3')
        parser.add_argument('--dino_model', type=str, default='vitl16', help='dino model type: vitl16 (for dinov3) or vitl14 (for dinov2)')

        parser.add_argument('--train_dir', type=str, default='datasets/AMBIENT6K/Train', help='dir of train data')
        parser.add_argument('--val_dir', type=str, default='datasets/AMBIENT6K/Test', help='dir of val data')
        parser.add_argument('--test_dir', type=str, default='', help='dir of test data with origin/depth/normal/shadow_free')
        parser.add_argument('--mask_dir', type=str, default='', help='dir of shadow masks; defaults to test_dir/shadow_mask')
        parser.add_argument('--warmup', action='store_true', default=True, help='warmup')
        parser.add_argument('--warmup_epochs', type=int, default=3, help='epochs for warmup')
        parser.add_argument("--local-rank", type=int, default=0)

        return parser
