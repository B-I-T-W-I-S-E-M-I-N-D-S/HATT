import os
import json
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import opts_thumos as opts
import time
import h5py
from tqdm import tqdm
from iou_utils import *
from eval import evaluation_detection
from tensorboardX import SummaryWriter
from dataset import VideoDataSet
from models import MYNET, SuppressNet
from loss_func import cls_loss_func, cls_loss_func_, regress_loss_func
from loss_func import MultiCrossEntropyLoss
from functools import *

def train_one_epoch(opt, model, train_dataset, optimizer, warmup=False):
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=opt['batch_size'], shuffle=True,
                                                num_workers=0, pin_memory=True, drop_last=False)      
    epoch_cost = 0
    epoch_cost_cls = 0
    epoch_cost_reg = 0
    epoch_cost_snip = 0
    
    total_iter = len(train_dataset)//opt['batch_size']
    cls_loss = MultiCrossEntropyLoss(num_classes=opt['num_of_class'], focal=True)
    snip_loss = MultiCrossEntropyLoss(num_classes=opt['num_of_class'], focal=True)
    
    valid_batches = 0  # Track valid batches to avoid division by zero
    
    for n_iter, (input_data, cls_label, reg_label, snip_label) in enumerate(tqdm(train_loader)):
        if warmup and total_iter > 0:
            for g in optimizer.param_groups:
                g['lr'] = n_iter * (opt['lr']) / total_iter
        
        try:
            input_data = input_data.cuda(non_blocking=True)
            cls_label = cls_label.cuda(non_blocking=True)
            reg_label = reg_label.cuda(non_blocking=True)
            snip_label = snip_label.cuda(non_blocking=True)

            act_cls, act_reg, snip_cls = model(input_data)

            # Check for NaN values in model outputs
            if torch.isnan(act_cls).any() or torch.isnan(act_reg).any() or torch.isnan(snip_cls).any():
                print(f"Warning: NaN detected in model outputs at iteration {n_iter}")
                continue
            
            act_cls.register_hook(partial(cls_loss.collect_grad, cls_label))
            snip_cls.register_hook(partial(snip_loss.collect_grad, snip_label))
            
            cost_reg = 0
            cost_cls = 0

            loss = cls_loss_func_(cls_loss, cls_label, act_cls)
            if not torch.isnan(loss) and not torch.isinf(loss):
                cost_cls = loss
                epoch_cost_cls += cost_cls.detach().cpu().numpy()
            else:
                print(f"Warning: Invalid cls loss at iteration {n_iter}")
                continue
                   
            loss = regress_loss_func(reg_label, act_reg)
            if not torch.isnan(loss) and not torch.isinf(loss):
                cost_reg = loss  
                epoch_cost_reg += cost_reg.detach().cpu().numpy()
            else:
                print(f"Warning: Invalid reg loss at iteration {n_iter}")
                continue

            loss = cls_loss_func_(snip_loss, snip_label, snip_cls)
            if not torch.isnan(loss) and not torch.isinf(loss):
                cost_snip = loss
                epoch_cost_snip += cost_snip.detach().cpu().numpy()
            else:
                print(f"Warning: Invalid snip loss at iteration {n_iter}")
                continue
            
            cost = opt['alpha']*cost_cls + opt['beta']*cost_reg + opt['gamma']*cost_snip    
            
            # Check for NaN in total cost
            if torch.isnan(cost) or torch.isinf(cost):
                print(f"Warning: Invalid total cost at iteration {n_iter}")
                continue
                    
            epoch_cost += cost.detach().cpu().numpy()
            valid_batches += 1

            optimizer.zero_grad()
            cost.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
        except Exception as e:
            print(f"Error in training iteration {n_iter}: {e}")
            continue
                
    # Avoid division by zero
    if valid_batches == 0:
        return 0, 0, 0, 0, 0
        
    return valid_batches, epoch_cost/valid_batches, epoch_cost_cls/valid_batches, epoch_cost_reg/valid_batches, epoch_cost_snip/valid_batches

def eval_one_epoch(opt, model, test_dataset):
    cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames = eval_frame(opt, model, test_dataset)
        
    result_dict = eval_map_nms(opt, test_dataset, output_cls, output_reg, labels_cls, labels_reg)
    output_dict = {"version":"VERSION 1.3","results":result_dict,"external_data":{}}
    outfile = open(opt["result_file"].format(opt['exp']),"w")
    json.dump(output_dict, outfile, indent=2)
    outfile.close()
    
    IoUmAP = evaluation_detection(opt, verbose=False)
    if IoUmAP is not None and len(IoUmAP) > 0:
        IoUmAP_5 = sum(IoUmAP[0:])/len(IoUmAP[0:])
    else:
        IoUmAP_5 = 0.0

    return cls_loss, reg_loss, tot_loss, IoUmAP_5

def train(opt): 
    writer = SummaryWriter()
    model = MYNET(opt)
    
    # Move model to GPU and wrap with DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.cuda()
    
    rest_of_model_params = [param for name, param in model.named_parameters() if "history_unit" not in name]
  
    optimizer = optim.Adam([{'params': model.history_unit.parameters(), 'lr': 1e-6}, {'params': rest_of_model_params}], lr=opt["lr"], weight_decay=opt["weight_decay"])  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["lr_step"])
    
    opt["split"] = "train"

    train_dataset = VideoDataSet(opt, subset="train")      
    test_dataset = VideoDataSet(opt, subset=opt['inference_subset'])
    
    warmup = False
    
    for n_epoch in range(opt['epoch']):   
        if n_epoch >= 1:
            warmup = False
        
        n_iter, epoch_cost, epoch_cost_cls, epoch_cost_reg, epoch_cost_snip = train_one_epoch(opt, model, train_dataset, optimizer, warmup)
        
        # Handle case where no valid iterations occurred
        if n_iter == 0:
            print(f"Warning: No valid training iterations in epoch {n_epoch}")
            continue
            
        writer.add_scalars('data/cost', {'train': epoch_cost}, n_epoch)
        print("training loss(epoch %d): %.03f, cls - %.06f, reg - %.06f, snip - %.06f, lr - %.06f"%(n_epoch,
                                                                            epoch_cost,
                                                                            epoch_cost_cls,
                                                                            epoch_cost_reg,
                                                                            epoch_cost_snip,
                                                                            optimizer.param_groups[-1]["lr"]) )
        
        scheduler.step()
        model.eval()
        
        cls_loss, reg_loss, tot_loss, IoUmAP_5 = eval_one_epoch(opt, model, test_dataset)
        
        writer.add_scalars('data/mAP', {'test': IoUmAP_5}, n_epoch)
        print("testing loss(epoch %d): %.06f, cls - %.06f, reg - %.06f, mAP Avg - %.06f"%(n_epoch, tot_loss, cls_loss, reg_loss, IoUmAP_5))
                    
        # Save the state_dict of the model (not DataParallel wrapper)
        state = {'epoch': n_epoch + 1,
                 'state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()}
        torch.save(state, opt["checkpoint_path"]+"/"+opt["exp"]+"_checkpoint_"+str(n_epoch+1)+".pth.tar")
        
        # Initialize best_map if it doesn't exist
        if not hasattr(model, 'best_map'):
            model.best_map = 0.0
            
        if IoUmAP_5 > model.best_map:
            model.best_map = IoUmAP_5
            torch.save(state, opt["checkpoint_path"]+"/"+opt["exp"]+"_ckp_best.pth.tar")
            
        model.train()
                
    writer.close()
    return getattr(model, 'best_map', 0.0)

def eval_frame(opt, model, dataset):
    test_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt['batch_size'], shuffle=False,
                                             num_workers=0, pin_memory=True, drop_last=False)
    
    labels_cls = {}
    labels_reg = {}
    output_cls = {}
    output_reg = {}                                      
    for video_name in dataset.video_list:
        labels_cls[video_name] = []
        labels_reg[video_name] = []
        output_cls[video_name] = []
        output_reg[video_name] = []
        
    start_time = time.time()
    total_frames = 0  
    epoch_cost = 0.0
    epoch_cost_cls = 0.0
    epoch_cost_reg = 0.0
    valid_batches = 0
    
    for n_iter, (input_data, cls_label, reg_label, _) in enumerate(tqdm(test_loader)):
        try:
            input_data = input_data.cuda(non_blocking=True)
            cls_label = cls_label.cuda(non_blocking=True)
            reg_label = reg_label.cuda(non_blocking=True)

            act_cls, act_reg, _ = model(input_data)
            
            # Check for NaN values
            if torch.isnan(act_cls).any() or torch.isnan(act_reg).any():
                print(f"Warning: NaN detected in eval outputs at iteration {n_iter}")
                continue
                
            cost_reg = 0
            cost_cls = 0
            
            loss = cls_loss_func(cls_label, act_cls)
            if not torch.isnan(loss) and not torch.isinf(loss):
                cost_cls = loss
                epoch_cost_cls += cost_cls.detach().cpu().numpy()
            else:
                continue
                   
            loss = regress_loss_func(reg_label, act_reg)
            if not torch.isnan(loss) and not torch.isinf(loss):
                cost_reg = loss  
                epoch_cost_reg += cost_reg.detach().cpu().numpy()
            else:
                continue
            
            cost = opt['alpha']*cost_cls + opt['beta']*cost_reg    
            
            if not torch.isnan(cost) and not torch.isinf(cost):
                epoch_cost += cost.detach().cpu().numpy()
                valid_batches += 1
            else:
                continue
            
            act_cls = torch.softmax(act_cls, dim=-1)
            
            total_frames += input_data.size(0)
            
            for b in range(0, input_data.size(0)):
                video_name, st, ed, data_idx = dataset.inputs[n_iter*opt['batch_size']+b]
                output_cls[video_name] += [act_cls[b,:].detach().cpu().numpy()]
                output_reg[video_name] += [act_reg[b,:].detach().cpu().numpy()]
                labels_cls[video_name] += [cls_label[b,:].numpy()]
                labels_reg[video_name] += [reg_label[b,:].numpy()]
                
        except Exception as e:
            print(f"Error in eval iteration {n_iter}: {e}")
            continue
        
    end_time = time.time()
    working_time = end_time - start_time
    
    for video_name in dataset.video_list:
        if len(labels_cls[video_name]) > 0:
            labels_cls[video_name] = np.stack(labels_cls[video_name], axis=0)
            labels_reg[video_name] = np.stack(labels_reg[video_name], axis=0)
            output_cls[video_name] = np.stack(output_cls[video_name], axis=0)
            output_reg[video_name] = np.stack(output_reg[video_name], axis=0)
    
    # Avoid division by zero
    if valid_batches == 0:
        cls_loss = 0.0
        reg_loss = 0.0
        tot_loss = 0.0
    else:
        cls_loss = epoch_cost_cls / valid_batches
        reg_loss = epoch_cost_reg / valid_batches
        tot_loss = epoch_cost / valid_batches
     
    return cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames

def eval_map_nms(opt, dataset, output_cls, output_reg, labels_cls, labels_reg):
    result_dict = {}
    proposal_dict = []
    
    num_class = opt["num_of_class"]
    unit_size = opt['segment_size']
    threshold = opt['threshold']
    anchors = opt['anchors']
    
    # Debug print to check label_name length
    print(f"Number of classes: {num_class}, Label name length: {len(dataset.label_name)}")
    print(f"Label names: {dataset.label_name}")
                                             
    for video_name in dataset.video_list:
        if video_name not in output_cls or len(output_cls[video_name]) == 0:
            result_dict[video_name] = []
            continue
            
        duration = dataset.video_len[video_name]
        video_time = float(dataset.video_dict[video_name]["duration"])
        frame_to_time = 100.0*video_time / duration
         
        for idx in range(0, min(duration, len(output_cls[video_name]))):
            cls_anc = output_cls[video_name][idx]
            reg_anc = output_reg[video_name][idx]
            
            proposal_anc_dict = []
            for anc_idx in range(0, len(anchors)):
                # Ensure we have valid data
                if anc_idx >= len(cls_anc) or anc_idx >= len(reg_anc):
                    continue
                    
                # Handle potential NaN values
                if np.isnan(cls_anc[anc_idx]).any() or np.isnan(reg_anc[anc_idx]).any():
                    continue
                
                # Make sure we don't go beyond the available classes
                max_class_idx = min(len(cls_anc[anc_idx]), len(dataset.label_name))
                
                # Use np.where instead of np.argwhere for better NaN handling
                valid_mask = ~np.isnan(cls_anc[anc_idx][:max_class_idx])
                threshold_mask = cls_anc[anc_idx][:max_class_idx] > threshold
                cls_indices = np.where(valid_mask & threshold_mask)[0]
                
                if len(cls_indices) == 0:
                    continue
                
                # Handle potential NaN in regression outputs
                if np.isnan(reg_anc[anc_idx]).any():
                    continue
                    
                ed = idx + anchors[anc_idx] * reg_anc[anc_idx][0]
                length = anchors[anc_idx] * np.exp(np.clip(reg_anc[anc_idx][1], -10, 10))  # Clip to prevent overflow
                st = ed - length
                
                for label in cls_indices:
                    # Additional safety check
                    if label >= len(dataset.label_name):
                        print(f"Warning: Label index {label} is out of range. Max valid index is {len(dataset.label_name)-1}")
                        continue
                        
                    score = cls_anc[anc_idx][label]
                    if np.isnan(score) or np.isinf(score):
                        continue
                        
                    tmp_dict = {}
                    tmp_dict["segment"] = [float(st*frame_to_time/100.0), float(ed*frame_to_time/100.0)]
                    tmp_dict["score"] = float(score)
                    tmp_dict["label"] = dataset.label_name[label]
                    tmp_dict["gentime"] = float(idx*frame_to_time/100.0)
                    proposal_anc_dict.append(tmp_dict)
                
            proposal_dict += proposal_anc_dict
        
        proposal_dict = non_max_suppression(proposal_dict, overlapThresh=opt['soft_nms'])
                    
        result_dict[video_name] = proposal_dict
        proposal_dict = []
        
    return result_dict

def eval_map_supnet(opt, dataset, output_cls, output_reg, labels_cls, labels_reg):
    model = SuppressNet(opt)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.cuda()
    checkpoint = torch.load(opt["checkpoint_path"]+"/ckp_best_suppress.pth.tar")
    base_dict = checkpoint['state_dict']
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(base_dict)
    else:
        model.load_state_dict(base_dict)
    model.eval()
    
    result_dict = {}
    proposal_dict = []
    
    num_class = opt["num_of_class"]
    unit_size = opt['segment_size']
    threshold = opt['threshold']
    anchors = opt['anchors']
                                             
    for video_name in dataset.video_list:
        if video_name not in output_cls or len(output_cls[video_name]) == 0:
            result_dict[video_name] = []
            continue
            
        duration = dataset.video_len[video_name]
        video_time = float(dataset.video_dict[video_name]["duration"])
        frame_to_time = 100.0*video_time / duration
        conf_queue = torch.zeros((unit_size, num_class-1)) 
        
        for idx in range(0, min(duration, len(output_cls[video_name]))):
            cls_anc = output_cls[video_name][idx]
            reg_anc = output_reg[video_name][idx]
            
            proposal_anc_dict = []
            for anc_idx in range(0, len(anchors)):
                if anc_idx >= len(cls_anc) or anc_idx >= len(reg_anc):
                    continue
                    
                if np.isnan(cls_anc[anc_idx]).any() or np.isnan(reg_anc[anc_idx]).any():
                    continue
                
                max_class_idx = min(len(cls_anc[anc_idx]), len(dataset.label_name))
                valid_mask = ~np.isnan(cls_anc[anc_idx][:max_class_idx])
                threshold_mask = cls_anc[anc_idx][:max_class_idx] > threshold
                cls_indices = np.where(valid_mask & threshold_mask)[0]
                
                if len(cls_indices) == 0:
                    continue
                
                if np.isnan(reg_anc[anc_idx]).any():
                    continue
                    
                ed = idx + anchors[anc_idx] * reg_anc[anc_idx][0]
                length = anchors[anc_idx] * np.exp(np.clip(reg_anc[anc_idx][1], -10, 10))
                st = ed - length
                
                for label in cls_indices:
                    if label >= len(dataset.label_name):
                        continue
                    
                    score = cls_anc[anc_idx][label]
                    if np.isnan(score) or np.isinf(score):
                        continue
                        
                    tmp_dict = {}
                    tmp_dict["segment"] = [float(st*frame_to_time/100.0), float(ed*frame_to_time/100.0)]
                    tmp_dict["score"] = float(score)
                    tmp_dict["label"] = dataset.label_name[label]
                    tmp_dict["gentime"] = float(idx*frame_to_time/100.0)
                    proposal_anc_dict.append(tmp_dict)
                          
            proposal_anc_dict = non_max_suppression(proposal_anc_dict, overlapThresh=opt['soft_nms'])  
                
            conf_queue[:-1,:] = conf_queue[1:,:].clone()
            conf_queue[-1,:] = 0
            for proposal in proposal_anc_dict:
                if proposal['label'] in dataset.label_name:
                    cls_idx = dataset.label_name.index(proposal['label'])
                    if cls_idx < conf_queue.shape[1]:
                        conf_queue[-1, cls_idx] = proposal["score"]
            
            minput = conf_queue.unsqueeze(0).cuda(non_blocking=True)
            suppress_conf = model(minput)
            suppress_conf = suppress_conf.squeeze(0).detach().cpu().numpy()
            
            for cls in range(0, min(num_class-1, len(suppress_conf))):
                if suppress_conf[cls] > opt['sup_threshold']:
                    for proposal in proposal_anc_dict:
                        if cls < len(dataset.label_name) and proposal['label'] == dataset.label_name[cls]:
                            if check_overlap_proposal(proposal_dict, proposal, overlapThresh=opt['soft_nms']) is None:
                                proposal_dict.append(proposal)
            
        result_dict[video_name] = proposal_dict
        proposal_dict = []
        
    return result_dict

def test_frame(opt): 
    model = MYNET(opt)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.cuda()
    checkpoint = torch.load(opt["checkpoint_path"]+"/ckp_best.pth.tar")
    base_dict = checkpoint['state_dict']
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(base_dict)
    else:
        model.load_state_dict(base_dict)
    model.eval()
    
    dataset = VideoDataSet(opt, subset=opt['inference_subset'])    
    outfile = h5py.File(opt['frame_result_file'].format(opt['exp']), 'w')
    
    cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames = eval_frame(opt, model, dataset)
    
    print("testing loss: %.06f, cls_loss: %.06f, reg_loss: %.06f"%(tot_loss, cls_loss, reg_loss))
    
    for video_name in dataset.video_list:
        if video_name in output_cls and len(output_cls[video_name]) > 0:
            o_cls = output_cls[video_name]
            o_reg = output_reg[video_name]
            l_cls = labels_cls[video_name]
            l_reg = labels_reg[video_name]
            
            dset_predcls = outfile.create_dataset(video_name+'/pred_cls', o_cls.shape, maxshape=o_cls.shape, chunks=True, dtype=np.float32)
            dset_predcls[:,:] = o_cls[:,:]  
            dset_predreg = outfile.create_dataset(video_name+'/pred_reg', o_reg.shape, maxshape=o_reg.shape, chunks=True, dtype=np.float32)
            dset_predreg[:,:] = o_reg[:,:]   
            dset_labelcls = outfile.create_dataset(video_name+'/label_cls', l_cls.shape, maxshape=l_cls.shape, chunks=True, dtype=np.float32)
            dset_labelcls[:,:] = l_cls[:,:]   
            dset_labelreg = outfile.create_dataset(video_name+'/label_reg', l_reg.shape, maxshape=l_reg.shape, chunks=True, dtype=np.float32)
            dset_labelreg[:,:] = l_reg[:,:]   
    outfile.close()
                    
    print("working time : {}s, {}fps, {} frames".format(working_time, total_frames/working_time if working_time > 0 else 0, total_frames))
    
def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return forward_orig(*args, **kwargs)

    m.forward = wrap

class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []

def test(opt): 
    model = MYNET(opt)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.cuda()
    checkpoint = torch.load(opt["checkpoint_path"]+"/"+opt['exp']+"_ckp_best.pth.tar")
    base_dict = checkpoint['state_dict']
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(base_dict)
    else:
        model.load_state_dict(base_dict)
    model.eval()
    
    opt["split"] = "test"
    dataset = VideoDataSet(opt, subset=opt['inference_subset'])
    
    cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames = eval_frame(opt, model, dataset)
    
    if opt["pptype"] == "nms":
        result_dict = eval_map_nms(opt, dataset, output_cls, output_reg, labels_cls, labels_reg)
    elif opt["pptype"] == "net":
        result_dict = eval_map_supnet(opt, dataset, output_cls, output_reg, labels_cls, labels_reg)
    else:
        result_dict = eval_map_nms(opt, dataset, output_cls, output_reg, labels_cls, labels_reg)
        
    output_dict = {"version":"VERSION 1.3","results":result_dict,"external_data":{}}
    outfile = open(opt["result_file"].format(opt['exp']),"w")
    json.dump(output_dict, outfile, indent=2)
    outfile.close()
    
    mAP = evaluation_detection(opt)

def test_online(opt): 
    model = MYNET(opt)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.cuda()
    checkpoint = torch.load(opt["checkpoint_path"]+"/ckp_best.pth.tar")
    base_dict = checkpoint['state_dict']
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(base_dict)
    else:
        model.load_state_dict(base_dict)
    model.eval()
    
    sup_model = SuppressNet(opt)
    if torch.cuda.device_count() > 1:
        sup_model = nn.DataParallel(sup_model)
    sup_model = sup_model.cuda()
    checkpoint = torch.load(opt["checkpoint_path"]+"/ckp_best_suppress.pth.tar")
    base_dict = checkpoint['state_dict']
    if isinstance(sup_model, nn.DataParallel):
        sup_model.module.load_state_dict(base_dict)
    else:
        sup_model.load_state_dict(base_dict)
    sup_model.eval()
    
    dataset = VideoDataSet(opt, subset=opt['inference_subset'])
    test_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1, shuffle=False,
                                             num_workers=0, pin_memory=True, drop_last=False)
    
    result_dict = {}
    proposal_dict = []
    
    num_class = opt["num_of_class"]
    unit_size = opt['segment_size']
    threshold = opt['threshold']
    anchors = opt['anchors']
    
    start_time = time.time()
    total_frames = 0 
    
    for video_name in dataset.video_list:
        input_queue = torch.zeros((unit_size, opt['feat_dim'])) 
        sup_queue = torch.zeros((unit_size, num_class-1))
    
        duration = dataset.video_len[video_name]
        video_time = float(dataset.video_dict[video_name]["duration"])
        frame_to_time = 100.0*video_time / duration
        
        for idx in range(0, duration):
            total_frames += 1
            input_queue[:-1,:] = input_queue[1:,:].clone()
            input_queue[-1:,:] = dataset._get_base_data(video_name, idx, idx+1)
            
            minput = input_queue.unsqueeze(0).cuda(non_blocking=True)
            act_cls, act_reg, _ = model(minput)
            act_cls = torch.softmax(act_cls, dim=-1)
            
            cls_anc = act_cls.squeeze(0).detach().cpu().numpy()
            reg_anc = act_reg.squeeze(0).detach().cpu().numpy()
            
            proposal_anc_dict = []
            for anc_idx in range(0, len(anchors)):
                if anc_idx >= len(cls_anc) or anc_idx >= len(reg_anc):
                    continue
                    
                if np.isnan(cls_anc[anc_idx]).any() or np.isnan(reg_anc[anc_idx]).any():
                    continue
                
                max_class_idx = min(len(cls_anc[anc_idx]), len(dataset.label_name))
                valid_mask = ~np.isnan(cls_anc[anc_idx][:max_class_idx])
                threshold_mask = cls_anc[anc_idx][:max_class_idx] > threshold
                cls_indices = np.where(valid_mask & threshold_mask)[0]
                
                if len(cls_indices) == 0:
                    continue
                
                if np.isnan(reg_anc[anc_idx]).any():
                    continue
                    
                ed = idx + anchors[anc_idx] * reg_anc[anc_idx][0]
                length = anchors[anc_idx] * np.exp(np.clip(reg_anc[anc_idx][1], -10, 10))
                st = ed - length
                
                for label in cls_indices:
                    if label >= len(dataset.label_name):
                        continue
                    
                    score = cls_anc[anc_idx][label]
                    if np.isnan(score) or np.isinf(score):
                        continue
                        
                    tmp_dict = {}
                    tmp_dict["segment"] = [float(st*frame_to_time/100.0), float(ed*frame_to_time/100.0)]
                    tmp_dict["score"] = float(cls_anc[anc_idx][label])
                    tmp_dict["label"] = dataset.label_name[label]
                    tmp_dict["gentime"] = float(idx*frame_to_time/100.0)
                    proposal_anc_dict.append(tmp_dict)
                          
            proposal_anc_dict = non_max_suppression(proposal_anc_dict, overlapThresh=opt['soft_nms'])  
                
            sup_queue[:-1,:]=sup_queue[1:,:].clone()
            sup_queue[-1,:]=0
            for proposal in proposal_anc_dict:
                cls_idx = dataset.label_name.index(proposal['label'])
                sup_queue[-1,cls_idx]=proposal["score"]
            
            minput = sup_queue.unsqueeze(0).cuda(non_blocking=True)
            suppress_conf = sup_model(minput)
            suppress_conf=suppress_conf.squeeze(0).detach().cpu().numpy()
            
            for cls in range(0, num_class-1):
                if suppress_conf[cls] > opt['sup_threshold']:
                    for proposal in proposal_anc_dict:
                        if proposal['label'] == dataset.label_name[cls]:
                            if check_overlap_proposal(proposal_dict, proposal, overlapThresh=opt['soft_nms']) is None:
                                proposal_dict.append(proposal)
            
        result_dict[video_name]=proposal_dict
        proposal_dict=[]
    
    end_time = time.time()
    working_time = end_time-start_time
    print("working time : {}s, {}fps, {} frames".format(working_time, total_frames/working_time, total_frames))
    
    output_dict={"version":"VERSION 1.3","results":result_dict,"external_data":{}}
    outfile=open(opt["result_file"].format(opt['exp']),"w")
    json.dump(output_dict, outfile, indent=2)
    outfile.close()
    
    evaluation_detection(opt)

def main(opt):
    max_perf=0
    if opt['mode'] == 'train':
        max_perf=train(opt)
    if opt['mode'] == 'test':
        test(opt)
    if opt['mode'] == 'test_frame':
        test_frame(opt)
    if opt['mode'] == 'test_online':
        test_online(opt)
    if opt['mode'] == 'eval':
        evaluation_detection(opt)
        
    return max_perf

if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"]) 
    opt_file=open(opt["checkpoint_path"]+"/"+opt["exp"]+"_opts.json","w")
    json.dump(opt, opt_file)
    opt_file.close()
    
    if opt['seed'] >= 0:
        seed = opt['seed'] 
        torch.manual_seed(seed)
        np.random.seed(seed)
           
    opt['anchors'] = [int(item) for item in opt['anchors'].split(',')]  
           
    main(opt)