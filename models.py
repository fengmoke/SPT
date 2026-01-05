import torch.nn as nn
import torch
from momentfm import MOMENTPipeline
from SPT import spt
import loralib as lora
# from mona import Mona
def get_model(cfg):
    if cfg.pre_trained_model =='classification':
        if cfg.finetune_type == 'Finetune_Full':
            model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-small", 
            model_kwargs={
                'task_name': 'classification', # Choose the task: One of ['reconstruction', 'forecasting', 'classification', 'embedding']
                'n_channels': cfg.n_channels, # number of input channels
                'num_class': cfg.num_class,
                'freeze_encoder': False, # Freeze the patch embedding layer
                'freeze_embedder': False, # Freeze the transformer encoder
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
        else:
            model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-small", 
            model_kwargs={
                'task_name': 'classification', # Choose the task: One of ['reconstruction', 'forecasting', 'classification', 'embedding']
                'n_channels': cfg.n_channels, # number of input channels
                'num_class': cfg.num_class,
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
    elif cfg.pre_trained_model =='embedding':
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-small", 
            model_kwargs={'task_name': 'embedding',
                          'freeze_encoder': True, # Freeze the patch embedding layer
                          'freeze_embedder': True, # Freeze the transformer encoder}, # We are loading the model in `embedding` mode
            }
        )
        
    model.init()
    if cfg.finetune_type == 'LoRA':
         # 添加LoRA适配器
        for name, module in model.named_modules():
            # 仅处理标准Linear层
            if isinstance(module, nn.Linear):
                if any(key in name for key in ['q', 'k', 'v', 'o']) and 'SelfAttention' in name:
                    # 替换注意力矩阵
                    new_layer = lora.Linear(
                        module.in_features,
                        module.out_features,
                        r=8,
                        lora_alpha=32
                    )
                    new_layer.weight = module.weight
                    parent = model.get_submodule(name.rsplit('.', 1)[0])
                    setattr(parent, name.split('.')[-1], new_layer)
                    
                # elif any(key in name for key in ['wi_0', 'wi_1', 'wo']) and 'DenseReluDense' in name:
                #     # 替换前馈网络层
                #     new_layer = lora.Linear(
                #         module.in_features,
                #         module.out_features,
                #         r=8,
                #         lora_alpha=32
                #     )
                #     new_layer.weight = module.weight
                #     parent = model.get_submodule(name.rsplit('.', 1)[0])
                #     setattr(parent, name.split('.')[-1], new_layer)

         # 冻结非LoRA参数
        lora.mark_only_lora_as_trainable(model)
        print(f'微调方式：{cfg.finetune_type}')
        
        # 保持分类头全参数可训练
        for param in model.head.linear.parameters():
            param.requires_grad = True
    if cfg.finetune_type == 'SPT_Full':
        encoder_layers =  model.encoder.block
        for i, layer in enumerate(encoder_layers):
            morf_module = spt(cfg, 1, 1, cfg.kernel_sizes,
                             routing_type=cfg.routing_type,
                             top_k=cfg.top_k)
            
            # Create custom forward that handles attention mask
            class MUSELayerWrapper(nn.Module):
                def __init__(self, morf, orig_layer):
                    super().__init__()
                    self.morf = morf
                    self.orig_layer = orig_layer
                    
                def forward(self, x, attention_mask=None, **kwargs):
                    # Add/remove channel dimension
                    x = x.unsqueeze(1)  # [B, 1, L, D]
                    x = self.morf(x)
                    x = x.squeeze(1)    # [B, L, D]
                    # Pass through original layer with mask
                    return self.orig_layer(x, attention_mask=attention_mask, **kwargs)
            
            new_layer = MUSELayerWrapper(morf_module, layer)
            
            # Replace the original layer
            model.encoder.block[i] = new_layer
    if cfg.finetune_type == 'Adapter_Full':
        encoder_layers =  model.encoder.block
        for i, layer in enumerate(encoder_layers):
            in_features = layer.layer[-1].DenseReluDense.wo.out_features
            Adapter_module = Adapter(in_features, cfg.alpha, reduction=4)
            
            # Create custom forward that handles attention mask
            class AdapterLayerWrapper(nn.Module):
                def __init__(self, adapter, orig_layer):
                    super().__init__()
                    self.adapter = adapter
                    self.orig_layer = orig_layer
                    
                def forward(self, x, attention_mask=None, **kwargs):
                    output  = self.orig_layer(x, attention_mask=attention_mask, **kwargs)
                    # 转换为列表进行修改
                    output_list = list(output)
                    output_list[0] = self.adapter(output_list[0])
                    return tuple(output_list)  # 重新转为元组
            
            new_layer = AdapterLayerWrapper(Adapter_module, layer)
            
            # Replace the original layer
            model.encoder.block[i] = new_layer
    #         model.encoder.block[i] = new_layer
    if cfg.finetune_type in ['Adapter']:
        # Get original head dimensions
        in_features = model.head.linear.in_features
        out_features = model.head.linear.out_features
        
        # Create custom head with mask handling
        class AdapterHead(nn.Module):
            def __init__(self, adapter, original_head):
                super().__init__()
                self.dropout = original_head.dropout
                self.adapter = adapter
                self.linear = original_head.linear
                
            def forward(self, x, input_mask=None):  # Add mask parameter
                x = torch.mean(x, dim=1)
                x = self.dropout(x)
                x = self.adapter(x)
                x = self.linear(x)
                return x
        
        # Replace head with adapter-enhanced version
        model.head = AdapterHead(
            Adapter(in_features, cfg.alpha, reduction=4),
            model.head
        )
    # print(model)
    return model


class Adapter(nn.Module):
    def __init__(self, c_in, alpha, reduction=4):
        super(Adapter, self).__init__()
        self.alpha = alpha
        self.fc = nn.Sequential(
            nn.Linear(c_in, 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(8, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        short_cut = x
        x = self.fc(x)
        if self.alpha is None:
            x = short_cut + x
        else:
            x = self.alpha * short_cut + (1 - self.alpha) * x
        return x

class Model_HAR(nn.Module):
    def __init__(self, cfg):
        super(Model_HAR, self).__init__()
        self.cfg = cfg
        self.feature_model = get_model(cfg)
        if self.cfg.finetune_type == 'NUSE':
            self.morf = spt(cfg, 1, 1, cfg.kernel_sizes, routing_type=cfg.routing_type, top_k=cfg.top_k)
        # if self.cfg.finetune_type == 'Mona':
        #     self.mona = Mona(1, 8)
        self.alpha = cfg.alpha

    def forward(self, x):
        if self.cfg.finetune_type == 'SPT':
            if self.cfg.mode == 'CAM':
                x = torch.squeeze(x, 1)
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
            x = self.morf(x)
            x = x.squeeze(1)
        if self.cfg.finetune_type == 'Mona':
            H, W = x.shape[1], x.shape[2]
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], 1)
            x = self.mona(x, (H, W))
            x = x.reshape(x.shape[0], H, W)
            # x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        x = self.feature_model(x_enc=x)
        if self.training:
            return x.logits
        else:
            return x.logits, x.embeddings
        # elif self.cfg.finetune_type == 'Adapter':
        #     x = x.permute(0, 2, 1)
        #     x = self.feature_model(x_enc=x)
        #     short_cut = x.embeddings
        #     x = self.adapter(x.embeddings)
        #     x = self.alpha * x + (1 - self.alpha) * short_cut
        #     x = self.fc(x)
        #     return x





def main():
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-small", 
        model_kwargs={
            'task_name': 'classification', # Choose the task: One of ['reconstruction', 'forecasting', 'classification', 'embedding']
            'n_channels': 3, # number of input channels
            'num_class': 17,
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
        # loc
        )
    model.init()
    print(model)

if __name__ == "__main__":
    main()