import torch
import time
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
import os
from tqdm import tqdm
from dataloader import *
from utils import *
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import argparse
from fvcore.nn import flop_count_table, flop_count_str, FlopCountAnalysis


torch.manual_seed(42)
torch.cuda.manual_seed(42)  
torch.backends.cudnn.deterministic = True  # 确保每次运行结果一致
torch.backends.cudnn.benchmark = False  # 禁用基准模式

def get_argparser():
    dataset_config = {'PAMAP2':[36, 12],
                      'MotionSense':[12, 6],
                      'HHAR':[12, 6],
                      }
    dataset_name = 'PAMAP22'
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='zero_shot', choices=['train', 'test', 'Deploy', 'zero_shot'])
    parser.add_argument('--finetune_type', type=str, default='SPT', choices=['base', 'LoRA', 'Adapter', 'Adapter_Full', 'SPT', 'SPT_Full', 'Finetune_Full'])
    parser.add_argument('--pre_trained_model', type=str, default='classification', choices=['classification', 'embedding'])
    parser.add_argument('--save_path', type=str, default='saved_models')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--ratio', type=bool, default=False)
    parser.add_argument('--attention', type=bool, default=True)
    parser.add_argument('--shot', type=str, default=None)
    parser.add_argument('--data', type=str, default=dataset_name)
    parser.add_argument('--n_channels', type=int, default=dataset_config[dataset_name][0])
    parser.add_argument('--num_class', type=int, default=dataset_config[dataset_name][1])
    parser.add_argument('--use_valid', type=bool, default=True)
    parser.add_argument('--splits', type=list, default=["train", "test"])
    parser.add_argument('--kernel_sizes', type=list, default=[3, 5, 7])
    parser.add_argument('--routing_type', type=str, default='soft', choices=['hard', 'soft'])
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default = 8)
    parser.add_argument('--workers', type=int, default=4)


    


    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    return parser.parse_args()



def train(args):
    model_save_path = get_save_path(args)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if args.shot is not None:
        split_data(args)
    train_loader, val_loader, test_loader = get_dataloaders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ==================== 3. 训练配置 ====================
    model = built_model(args)
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # 每5个epoch学习率降为原来的0.1倍
    num_epochs = args.epochs
    best_acc = 0.0
    # if os.path.exists(os.path.join(model_save_path, 'best_model.pth')):
    #     best_acc = torch.load(os.path.join(model_save_path, 'best_model.pth'))['acc']
    # ==================== 4. 训练循环 ====================
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs = inputs.permute(0, 2, 1)  
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录统计信息
            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix({'loss': loss.item()})
        
        # # 更新学习率
        # scheduler.step() 
        
        # 计算epoch平均损失
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1} Loss: {epoch_loss:.4f}')

        # ========== 验证阶段 (示例) ==========
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs_val, labels_val in test_loader:  # 实际使用时替换为验证集
                inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                # inputs_val = inputs_val.permute(0, 2, 1)
                outputs_val = model(inputs_val)
                _, preds = torch.max(outputs_val, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_val.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        print(f'Validation Accuracy: {acc:.4f}')
        
        # ========== 保存最佳模型 ==========
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'acc': acc
            }, os.path.join(model_save_path, 'best_model.pth'))
            print(f'New best model saved with accuracy {acc:.4f}')

    # ==================== 5. 测试和部署 ====================
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(model_save_path, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:  # 实际使用时替换为验证集
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs = inputs.permute(0, 2, 1)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
    acc = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {acc:.4f}')


def test(args):
    Suffix = ''
    if args.finetune_type == 'MoRF':
        Suffix += '_attention' if args.attention else '_no_attention'
    if args.shot is not None:
        split_data(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders(args)
    model = built_model(args)
    model.to(device)
    model_save_path = get_save_path(args)
    checkpoint = torch.load(os.path.join(model_save_path, f'best_model{Suffix}.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    all_preds = []
    all_labels = []
    features = []
    test_bar = tqdm(test_loader)
    with torch.no_grad():
        for inputs, labels in test_bar:  # 实际使用时替换为验证集
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs = inputs.permute(0, 2, 1)
            outputs, embdeding = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            features.append(embdeding.cpu().numpy())
        
    acc = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {acc:.4f}')
    features_save_path = os.path.join('T-SNE', args.data)
    if not os.path.exists(features_save_path):
        os.makedirs(features_save_path)
    np.save(os.path.join(features_save_path, f'features_{args.finetune_type}.npy'), np.concatenate(features, axis=0))
    np.save(os.path.join(features_save_path, f'labels_{args.finetune_type}.npy'), np.array(all_labels))
    # input = torch.randn((1, 512, args.n_channels), device=device)
    # # analyze = FlopCountAnalysis(model, (input,))
    # # print(flop_count_str(analyze))
    # total_flops = FlopCountAnalysis(model, (input,)).total()
    # print(f"Total FLOPs: {total_flops / 1e9:.2f} G")  # 转换为GFLOPs显示



def zero_shot(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders(args)
    model = model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-small", 
            model_kwargs={
                'task_name': 'classification', # Choose the task: One of ['reconstruction', 'forecasting', 'classification', 'embedding']
                'n_channels': args.n_channels, # number of input channels
                'num_class': args.num_class,
                'freeze_encoder': True, # Freeze the patch embedding layer
                'freeze_embedder': True, # Freeze the transformer encoder
                'freeze_head': False, # The linear forecasting head must be trained
                ## NOTE: Disable gradient checkpointing to supress the warning when linear probing the model as MOMENT encoder is frozen
                'enable_gradient_checkpointing': False,
                # Choose how embedding is obtained from the model: One of ['mean', 'concat']
                # Multi-channel embeddings are obtained by either averaging or concatenating patch embeddings 
                # along the channel dimension. 'concat' results in embeddings of size (n_channels * d_model), 
                # while 'mean' results in embeddings of size (d_model)
                'reduction': 'concat',
            },
            local_files_only=True
            # loc
        )
    model.init()
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    features = []
    test_bar = tqdm(test_loader)
    with torch.no_grad():
        for inputs, labels in test_bar:  # 实际使用时替换为验证集
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.permute(0, 2, 1)
            outputs = model(x_enc=inputs)
            _, preds = torch.max(outputs.logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            features.append(outputs.embeddings.cpu().numpy())
        
    acc = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {acc:.4f}')
    features_save_path = os.path.join('T-SNE', args.data)
    if not os.path.exists(features_save_path):
        os.makedirs(features_save_path)
    np.save(os.path.join(features_save_path, f'features_zero_shot.pth'), np.concatenate(features, axis=0))
    np.save(os.path.join(features_save_path, f'labels_zero_shot.pth'), np.array(all_labels))


def Deploy(args):
    Suffix = ''
    if args.finetune_type == 'MoRF':
        Suffix += '_attention' if args.attention else '_no_attention'
    args.batch_size = 1
    train_loader, val_loader, test_loader = get_dataloaders(args)
    test_bar = tqdm(test_loader)
    model = built_model(args)
    model_save_path = get_save_path(args)
    checkpoint = torch.load(os.path.join(model_save_path, f'best_model{Suffix}.pth'), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    all_preds = []
    all_labels = []
    deploy_time = []
    with torch.no_grad():
        for inputs, labels in test_bar:  # 实际使用时替换为验证集
            # inputs = inputs.permute(0, 2, 1)
            t0 = time.time()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            t1 = time.time()
            all_preds.extend(preds.detach().numpy())
            all_labels.extend(labels.numpy())
            deploy_time.append((t1 - t0) * 1000)
        
    acc = accuracy_score(all_labels, all_preds)
    np.save(f'deploy/{args.data}_deploy_time.npy',np.array(deploy_time))
    print(f"Mean inference time: {sum(deploy_time)/len(deploy_time):.2f}毫秒")
    print(f'Test Accuracy: {acc:.4f}')




    
if __name__ == '__main__':
    args = get_argparser()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'Deploy':
        Deploy(args)
    elif args.mode == 'zero_shot':
        zero_shot(args)