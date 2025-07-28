import torch.nn.parallel
import torch.utils.data.distributed
import os
import shutil
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather

from sklearn.metrics import mean_absolute_error, r2_score
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr


def calculate_mape_ignore_zeros(y_true, y_pred):
    mask = y_true != 0
    if np.any(mask):
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
        return mape
    return None

def calculate_re(y_true, y_pred):
    mask = y_true != 0
    if np.any(mask):
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        return np.sqrt(np.sum((y_true_filtered - y_pred_filtered) ** 2)) / np.sqrt(np.sum(y_true_filtered ** 2))
    return None

def calculate_rae(y_true, y_pred):
    mask = y_true != 0
    if np.any(mask):
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        return np.sum(np.abs(y_true_filtered - y_pred_filtered)) / np.sum(np.abs(y_true_filtered))
    return None

def calculate_me(y_true, y_pred):
    return np.mean(y_pred - y_true)

def calculate_mrae(y_true, y_pred):
    mask = y_true != 0
    if np.any(mask):
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        return np.mean(np.abs(y_true_filtered - y_pred_filtered) / np.abs(y_true_filtered))
    return None

def calculate_nrmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    nrmse = np.sqrt(mse) / (np.max(y_true) - np.min(y_true))
    return nrmse


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args, logger):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    run_mape = AverageMeter()
    run_mae = AverageMeter()
    run_r2 = AverageMeter()
    run_me = AverageMeter()
    run_psnr = AverageMeter()
    run_re = AverageMeter()
    run_rae = AverageMeter()
    run_mrae = AverageMeter()
    run_nrmse = AverageMeter()

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        logger.info("Using CPU as no GPU is available")

    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"], batch_data["label"]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)

        logits_np = logits.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        for logit, tgt in zip(logits_np, target_np):
            print(f"target: {np.unique(tgt.flatten())}")
            print(f"logit: {np.unique(logit.flatten())}")
            #logger.info(
            #    f"target: {np.unique(tgt.flatten())}\n"
            #    f"logit: {np.unique(logit.flatten())}\n"
            #)
            if calculate_mape_ignore_zeros(tgt.flatten(), logit.flatten()) is not None:
                run_mape.update(calculate_mape_ignore_zeros(tgt.flatten(), logit.flatten()))
            run_mae.update(mean_absolute_error(tgt.flatten(), logit.flatten()))
            if len(np.unique(tgt.flatten())) != 1:
                run_r2.update(r2_score(tgt.flatten(), logit.flatten()))
            run_me.update(calculate_me(tgt.flatten(), logit.flatten()))
            run_psnr.update(psnr(tgt, logit, data_range=logit.max() - logit.min()))
            if calculate_re(tgt.flatten(), logit.flatten()) is not None:
                run_re.update(calculate_re(tgt.flatten(), logit.flatten()))
            if calculate_rae(tgt.flatten(), logit.flatten()) is not None:
                run_rae.update(calculate_rae(tgt.flatten(), logit.flatten()))
            if calculate_mrae(tgt.flatten(), logit.flatten()) is not None:
                run_mrae.update(calculate_mrae(tgt.flatten(), logit.flatten()))
            run_nrmse.update(calculate_nrmse(tgt.flatten(), logit.flatten()))

        if args.rank == 0:
            logger.info(
                f"Epoch {epoch + 1}/{args.max_epochs} {idx + 1}/{len(loader)}\t"
                f"RMSE: {np.sqrt(run_loss.avg):.4f}\t"
                f"MAE: {run_mae.avg:.4f}\tR2: {run_r2.avg:.4f}\tME: {run_me.avg:.4f}\t"
                f"PSNR: {run_psnr.avg:.4f}\tRE: {run_re.avg:.4f}\tRAE: {run_rae.avg:.4f}\t"
                f"NRMSE: {run_nrmse.avg:.4f}\t"
                f"time {time.time() - start_time:.2f}s"
            )
        start_time = time.time()

    return np.sqrt(run_loss.avg), run_mape.avg, run_mae.avg, run_r2.avg, run_me.avg, 0, run_psnr.avg, run_re.avg, run_rae.avg, run_mrae.avg, run_nrmse.avg


def val_epoch(model, loader, epoch, args, logger, model_inferer=None):
    model.eval()
    run_rmse = AverageMeter()
    run_mape = AverageMeter()
    run_mae = AverageMeter()
    run_r2 = AverageMeter()
    run_me = AverageMeter()
    run_psnr = AverageMeter()
    run_re = AverageMeter()
    run_rae = AverageMeter()
    run_mrae = AverageMeter()
    run_nrmse = AverageMeter()
    start_time = time.time()

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        logger.info("Using CPU as no GPU is available")

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"], batch_data["label"]
            data, target = data.to(device), target.to(device)
            with autocast(enabled=args.amp):
                # logits = model(data)
                logits = model_inferer(data)

            logits_np = logits.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()

            logits_tensor = torch.tensor(logits_np).to(device)
            target_tensor = torch.tensor(target_np).to(device)

            mse = torch.nn.functional.mse_loss(logits_tensor, target_tensor, reduction='mean').item()
            rmse = np.sqrt(mse)

            if args.distributed:
                rmse_list = distributed_all_gather([rmse], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
                run_rmse.update(np.mean(np.stack(rmse_list, axis=0), axis=0), n=1)
            else:
                run_rmse.update(rmse, n=1)

            for logit, tgt in zip(logits_np, target_np):
                # print(f"target: {np.unique(tgt.flatten())}")
                # print(f"logit: {np.unique(logit.flatten())}")
                logger.info(
                        f"target: {np.unique(tgt.flatten())}\n"
                        f"logit: {np.unique(logit.flatten())}\n"
                    )

                if calculate_mape_ignore_zeros(tgt.flatten(), logit.flatten()) is not None:
                    run_mape.update(calculate_mape_ignore_zeros(tgt.flatten(), logit.flatten()))
                run_mae.update(mean_absolute_error(tgt.flatten(), logit.flatten()))
                if len(np.unique(tgt.flatten())) != 1:
                    run_r2.update(r2_score(tgt.flatten(), logit.flatten()))
                run_me.update(calculate_me(tgt.flatten(), logit.flatten()))
                run_psnr.update(psnr(tgt, logit, data_range=logit.max() - logit.min()))
                if calculate_re(tgt.flatten(), logit.flatten()) is not None:
                    run_re.update(calculate_re(tgt.flatten(), logit.flatten()))
                if calculate_rae(tgt.flatten(), logit.flatten()) is not None:
                    run_rae.update(calculate_rae(tgt.flatten(), logit.flatten()))
                if calculate_mrae(tgt.flatten(), logit.flatten()) is not None:
                    run_mrae.update(calculate_mrae(tgt.flatten(), logit.flatten()))
                run_nrmse.update(calculate_nrmse(tgt.flatten(), logit.flatten()))

            if args.rank == 0:
                avg_rmse = np.mean(run_rmse.avg)
                logger.info(
                    f"Val {epoch + 1}/{args.max_epochs} {idx + 1}/{len(loader)}\t"
                    f"RMSE: {avg_rmse:.4f}\t"
                    f"MAE: {run_mae.avg:.4f}\tR2: {run_r2.avg:.4f}\tME: {run_me.avg:.4f}\t"
                    f"PSNR: {run_psnr.avg:.4f}\tRE: {run_re.avg:.4f}\tRAE: {run_rae.avg:.4f}\t"
                    f"NRMSE: {run_nrmse.avg:.4f}\t"
                    f"time {time.time() - start_time:.2f}s"
                )
            start_time = time.time()

    return run_rmse.avg, run_mape.avg, run_mae.avg, run_r2.avg, run_me.avg, 0, run_psnr.avg, run_re.avg, run_rae.avg, run_mrae.avg, run_nrmse.avg


def save_checkpoint(model, epoch, args, logger, filename="model.pt", best_rmse=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_rmse": best_rmse, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    logger.info(f"Saving checkpoint {filename}")


def run_training(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_func,
        args,
        logger,
        model_inferer=None,
        scheduler=None,
        start_epoch=0,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        logger.info(f"Writing Tensorboard logs to {args.logdir}")
    scaler = None
    if args.amp:
        scaler = GradScaler()
    best_val_rmse = float('inf')

    for epoch in tqdm(range(start_epoch, args.max_epochs)):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        logger.info(f'{args.rank}\t{time.ctime()}\tEpoch: {epoch + 1}')
        epoch_time = time.time()

        train_rmse, train_mape, train_mae, train_r2, train_me, _, train_psnr, train_re, train_rae, train_mrae, train_nrmse = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args, logger=logger
        )

        if args.rank == 0:
            logger.info(
                f"Final training  {epoch + 1}/{args.max_epochs}\t"
                f"RMSE: {train_rmse:.4f}\t"
                f"MAE: {train_mae:.4f}\tR2: {train_r2:.4f}\tME: {train_me:.4f}\t"
                f"PSNR: {train_psnr:.4f}\tRE: {train_re:.4f}\tRAE: {train_rae:.4f}\t"
                f"NRMSE: {train_nrmse:.4f}\t"
                f"time {time.time() - epoch_time:.2f}s"
            )

        if writer is not None:
            writer.add_scalar("train_rmse", train_rmse, epoch)
            writer.add_scalar("train_mape", train_mape, epoch)
            writer.add_scalar("train_mae", train_mae, epoch)
            writer.add_scalar("train_r2", train_r2, epoch)
            writer.add_scalar("train_me", train_me, epoch)
            writer.add_scalar("train_psnr", train_psnr, epoch)
            writer.add_scalar("train_re", train_re, epoch)
            writer.add_scalar("train_rae", train_rae, epoch)
            writer.add_scalar("train_mrae", train_mrae, epoch)
            writer.add_scalar("train_nrmse", train_nrmse, epoch)

        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()

            val_avg_rmse, val_mape, val_mae, val_r2, val_me, _, val_psnr, val_re, val_rae, val_mrae, val_nrmse = val_epoch(
                model, val_loader, epoch=epoch, args=args, logger=logger, model_inferer=model_inferer
            )

            val_avg_rmse = np.mean(val_avg_rmse)

            if args.rank == 0:
                logger.info(
                    f"Final validation  {epoch + 1}/{args.max_epochs}\t"
                    f"RMSE: {val_avg_rmse:.4f}\tMAPE: {val_mape:.4f}\t"
                    f"MAE: {val_mae:.4f}\tR2: {val_r2:.4f}\tME: {val_me:.4f}\t"
                    f"PSNR: {val_psnr:.4f}\tRE: {val_re:.4f}\tRAE: {val_rae:.4f}\t"
                    f"MRAE: {val_mrae:.4f}\tNRMSE: {val_nrmse:.4f}\t"
                    f"time {time.time() - epoch_time:.2f}s"
                )

                if writer is not None:
                    writer.add_scalar("val_rmse", val_avg_rmse, epoch)
                    writer.add_scalar("val_mape", val_mape, epoch)
                    writer.add_scalar("val_mae", val_mae, epoch)
                    writer.add_scalar("val_r2", val_r2, epoch)
                    writer.add_scalar("val_me", val_me, epoch)
                    writer.add_scalar("val_psnr", val_psnr, epoch)
                    writer.add_scalar("val_re", val_re, epoch)
                    writer.add_scalar("val_rae", val_rae, epoch)
                    writer.add_scalar("val_mrae", val_mrae, epoch)
                    writer.add_scalar("val_nrmse", val_nrmse, epoch)

                if val_avg_rmse < best_val_rmse:
                    logger.info("new best RMSE ({:.6f} --> {:.6f}). ".format(best_val_rmse, val_avg_rmse))
                    best_val_rmse = val_avg_rmse
                    b_new_best = True
                    if args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, logger, best_rmse=best_val_rmse, optimizer=optimizer, scheduler=scheduler
                        )

            if args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, logger, best_rmse=best_val_rmse, filename="model_final.pt")
                if b_new_best:
                    logger.info("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    logger.info(f"Training Finished !, Best RMSE: {best_val_rmse}")

    return best_val_rmse

