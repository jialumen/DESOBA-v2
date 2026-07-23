import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import utils
import random
from model.sr3_modules import transformer


def run_validation(diffusion, val_loader, opt, current_epoch, current_step,
                   logger, tb_logger, wandb_logger=None, val_step=0,
                   model_restoration=None):
    idx = 0
    region_metrics = {key: [] for key in Metrics.REGION_METRIC_KEYS}
    mask_stats = []
    result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
    os.makedirs(result_path, exist_ok=True)

    metric_size = int(opt['train'].get('metric_size', 256) or 256)
    save_val_images = bool(opt['train'].get('save_val_images', True))
    save_val_image_limit = int(opt['train'].get('save_val_image_limit', 0) or 0)
    use_degradation = opt.get('setting') and opt['setting'].get('use_degradation_estimate')
    compose_with_input = bool(
        opt.get('setting') and opt['setting'].get('compose_output_with_input_eval', False))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    for _, val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        if use_degradation and model_restoration is not None:
            x_hat = model_restoration((diffusion.data['SR'] + 1) / 2, diffusion.data['mask'])
            x_hat = torch.clamp(x_hat, 0, 1)
            h_hat = (diffusion.data['SR'] + 1) / (2 * x_hat + 1e-4)
            h_hat = torch.where(h_hat == 0, h_hat + 1e-4, h_hat)
            diffusion.test_d(h_hat, continous=False)
        else:
            diffusion.test(continous=False)

        visuals = diffusion.get_current_visuals()
        metric_sr = visuals['SR']
        if compose_with_input:
            input_sr = diffusion.data['SR'].detach().float().cpu()
            mask = diffusion.data['mask'].detach().float().cpu()
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            metric_sr = metric_sr * mask + input_sr * (1.0 - mask)

        batch_metrics = Metrics.calculate_region_metrics_from_tensors(
            metric_sr, visuals['HR'], diffusion.data['mask'].detach().float().cpu(),
            size=metric_size)
        Metrics.merge_metric_lists(region_metrics, batch_metrics)
        mask_stats.extend(Metrics.calculate_mask_stats_from_tensors(
            diffusion.data['mask'].detach().float().cpu(), size=metric_size))

        should_save = save_val_images and (save_val_image_limit <= 0 or idx <= save_val_image_limit)
        if should_save:
            sr_img = Metrics.tensor2img(metric_sr)
            hr_img = Metrics.tensor2img(visuals['HR'])
            lr_img = Metrics.tensor2img(visuals['LR'])
            fake_img = Metrics.tensor2img(visuals['INF'], min_max=(0, 1))
            Metrics.save_img(
                hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
            tb_logger.add_image(
                'Iter_{}'.format(current_step),
                np.transpose(np.concatenate((sr_img, hr_img), axis=1), [2, 0, 1]),
                idx)

            if wandb_logger:
                wandb_logger.log_image(
                    f'validation_{idx}',
                    np.concatenate((sr_img, hr_img), axis=1))

    avg_metrics = Metrics.mean_metric_lists(region_metrics)
    avg_mask_stats = Metrics.mean_mask_stats(mask_stats)
    if opt['phase'] == 'train':
        diffusion.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')

    metric_text = Metrics.format_region_metrics(avg_metrics)
    mask_text = (
        '  Mask: raw_min {:.6f} | raw_max {:.6f} | raw_mean {:.6f} | eval_shadow_ratio {:.6f}'.format(
            avg_mask_stats['raw_min'], avg_mask_stats['raw_max'],
            avg_mask_stats['raw_mean'], avg_mask_stats['eval_shadow_ratio']))
    output_text = '  Output: compose_with_input {}'.format(compose_with_input)
    message = '[Validation][Epoch {} Iter {}]\n{}\n{}\n{}'.format(
        current_epoch, current_step, metric_text, mask_text, output_text)
    logger.info(message)
    logging.getLogger('val').info(message.replace('\n', ' '))

    for key, value in avg_metrics.items():
        tb_logger.add_scalar('validation/{}'.format(key), value, current_step)

    if wandb_logger:
        wandb_metrics = {'validation/{}'.format(k): float(v) for k, v in avg_metrics.items()}
        wandb_metrics['validation/val_step'] = val_step
        wandb_logger.log_metrics(wandb_metrics)
        val_step += 1
    return avg_metrics, val_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    val_step = 0
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
    else:
        wandb_logger = None

    # ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    if opt['setting']['use_degradation_estimate']:
        # degradation predict model
        model_restoration = transformer.Uformer()
        model_restoration.cuda()
        utils.load_checkpoint(model_restoration, opt['setting']['degradation_model_path'])
        model_restoration.eval()

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        early_stop_opt = opt['train'].get('early_stop') or {}
        early_stop_enabled = bool(early_stop_opt.get('enabled', False))
        early_stop_metric = early_stop_opt.get('metric', 'psnr_all')
        early_stop_mode = early_stop_opt.get('mode', 'max')
        early_stop_patience = int(early_stop_opt.get('patience', 8) or 8)
        early_stop_min_delta = float(early_stop_opt.get('min_delta', 0.01) or 0.0)
        early_stop_state = {
            'best': -float('inf') if early_stop_mode == 'max' else float('inf'),
            'bad_count': 0,
        }
        stop_training = False

        def update_early_stop(avg_metrics):
            if not early_stop_enabled:
                return False
            value = avg_metrics.get(early_stop_metric)
            if value is None or not np.isfinite(value):
                logger.info('Early stop metric [{}] is unavailable; continuing.'.format(
                    early_stop_metric))
                return False
            if early_stop_mode == 'min':
                improved = value < early_stop_state['best'] - early_stop_min_delta
            else:
                improved = value > early_stop_state['best'] + early_stop_min_delta
            if improved:
                early_stop_state['best'] = value
                early_stop_state['bad_count'] = 0
                logger.info('Early stop monitor improved: {}={:.6f}'.format(
                    early_stop_metric, value))
                return False
            early_stop_state['bad_count'] += 1
            logger.info('Early stop monitor no improvement: {}={:.6f}, best={:.6f}, bad_count={}/{}'.format(
                early_stop_metric, value, early_stop_state['best'],
                early_stop_state['bad_count'],
                early_stop_patience))
            return early_stop_state['bad_count'] >= early_stop_patience

        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                # if current_epoch > 5:
                #     target, input_, mask = utils.MixUp_AUG().aug(train_data['HR'].cuda(), train_data['SR'].cuda(), train_data['mask'].cuda())
                #     train_data['HR'] = target
                #     train_data['SR'] = input_
                #     train_data['mask'] = mask
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # Optional step-based validation for old configs.
                val_freq = int(opt['train'].get('val_freq', 0) or 0)
                if val_freq > 0 and current_step % val_freq == 0:
                    avg_metrics, val_step = run_validation(
                        diffusion, val_loader, opt, current_epoch, current_step,
                        logger, tb_logger, wandb_logger, val_step,
                        model_restoration if opt['setting']['use_degradation_estimate'] else None)
                    if update_early_stop(avg_metrics):
                        logger.info('Early stopping triggered at epoch {}, iter {}.'.format(
                            current_epoch, current_step))
                        stop_training = True
                        break

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            val_epoch_freq = int(opt['train'].get('val_epoch_freq', 0) or 0)
            if val_epoch_freq > 0 and current_epoch % val_epoch_freq == 0:
                avg_metrics, val_step = run_validation(
                    diffusion, val_loader, opt, current_epoch, current_step,
                    logger, tb_logger, wandb_logger, val_step,
                    model_restoration if opt['setting']['use_degradation_estimate'] else None)
                if update_early_stop(avg_metrics):
                    logger.info('Early stopping triggered at epoch {}, iter {}.'.format(
                        current_epoch, current_step))
                    stop_training = True

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

            if stop_training:
                break

        # save model
        if current_step > 0:
            logger.info('Saving final models and training states.')
            diffusion.save_network(current_epoch, current_step)
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        run_validation(
            diffusion, val_loader, opt, current_epoch, current_step,
            logger, tb_logger, wandb_logger, val_step,
            model_restoration if opt['setting']['use_degradation_estimate'] else None)
        raise SystemExit(0)
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            if opt['setting']['use_degradation_estimate']:
                x_hat = model_restoration((val_data['SR'] + 1) / 2, val_data['mask'])
                x_hat = torch.clamp(x_hat, 0, 1)
                # x_hat = x_hat * 2 - 1
                h_hat = (val_data['SR']+1) / (2 *(x_hat) + 1e-4)
                # h_hat = torch.clamp(h_hat, 0, 1)
                h_hat = torch.where(h_hat == 0, h_hat + 1e-4, h_hat)
                diffusion.test_d(h_hat, continous=True)
            else:
                diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
            fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                Metrics.save_img(
                    sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

            Metrics.save_img(
                hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

            # generation
            res = Metrics.tensor2img(visuals['SR'][-1])
            avg_channel = np.mean(res, axis=(0, 1))
            avg_channel_gt = np.mean(hr_img, axis=(0, 1))
            # res = res * avg_channel_gt / avg_channel
            eval_psnr = Metrics.calculate_psnr(res, hr_img)
            eval_ssim = Metrics.calculate_ssim(res, hr_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim
            print(f"ID: {idx}; PSNR: {eval_psnr}; SSIM: {eval_ssim}")

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            })
