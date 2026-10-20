"""Parameter-matched engineering controls for original SlotSPE diagnostics.

This is not a new research method. The original desktop source is read only.
Cross-modal attention deliberately remains dense: its use of all slots is an
existing design choice, not evidence that complementary slots were deleted.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F

SOURCE = Path(r'C:\Users\栋栋\Desktop\特征解耦论文\SlotSPE-main\SlotSPE-main')
sys.path.insert(0, str(SOURCE))
from models.SlotSPE import SlotSPE, MoESlotDecoder
from models.slot_attention import MultiHeadSlotAttention, gumbel_topk_st, parallel_topk_st


def make_args(omic_sizes, encoding_dim=1024):
    return SimpleNamespace(omic_sizes=list(omic_sizes), n_classes=4,
        encoding_dim=encoding_dim, wsi_projection_dim=256, rna_format='Pathways',
        slot_num_wsi=8, slot_num_omics=8, slot_iters=10, temperature=0.01,
        topk_ratio=0.25, top_k_method='parallel_topk_st', bag_loss='nll_surv',
        alpha_surv=0.5, lambda_recon_loss=0.01)


class ReproducibleSlotAttention(MultiHeadSlotAttention):
    def forward(self, inputs, num_slots=None):
        if self.training:
            return super().forward(inputs, num_slots=num_slots)
        b,n,d = inputs.shape
        n_s = self.num_slots if num_slots is None else num_slots
        if n_s != self._proof_eval_noise.shape[1]:
            raise ValueError('Diagnostic eval requires the configured slot count.')
        # Fixed distinct noise samples preserve the learned location and scale.
        # They do not depend on batch index, patient identity, or the global RNG.
        slots = self.slots_mu + self.slots_logsigma.exp() * self._proof_eval_noise.to(inputs)
        slots = slots.expand(b,-1,-1)
        normalized = self.norm_input(inputs)
        k,v = map(self.split_heads,(self.to_k(normalized),self.to_v(normalized)))
        for _ in range(self.iters):
            previous = slots
            q = self.split_heads(self.to_q(self.norm_slots(slots)))
            dots = torch.einsum('... i d, ... j d -> ... i j',q,k)*self.scale
            attn = F.normalize(dots.softmax(dim=-2)+self.eps,p=1,dim=-1)
            updates = torch.einsum('... j d, ... i j -> ... i d',v,attn)
            updates = self.combine_heads(self.merge_heads(updates))
            slots = self.gru(updates.reshape(-1,d),previous.reshape(-1,d)).reshape(b,n_s,d)
            slots = slots+self.mlp(self.norm_pre_ff(slots))
        return slots


class StableSlotDecoder(MoESlotDecoder):
    def forward(self, slots):
        mapped = self.map(slots)
        logits = self.decoder(mapped)
        scores = self.pred_keep_slot(mapped).squeeze(-1)
        if self.training:
            choose = gumbel_topk_st if self.top_k_method == 'gumbel_topk_st' else parallel_topk_st
            sampled,_ = choose(scores,k=self.k,temperature=self.temperature)
            # The original .01 temperature can give exactly zero numerical
            # derivatives. Retain the same sampled hard forward selection but
            # use a separately specified, nonsaturating gradient estimator.
            # This is an engineering control, not a claimed unbiased gradient.
            surrogate=self.k*(scores/1.0).softmax(-1)
            keep=sampled.detach()+(surrogate-surrogate.detach())
        else:
            indices = scores.topk(self.k,dim=-1).indices
            keep = torch.zeros_like(scores).scatter_(1,indices,1.)
        # Stable softmax on the selected support avoids tiny-mass / eps division.
        hard = keep.detach()>0.5
        gate = (scores/self.temperature).masked_fill(~hard,float('-inf')).softmax(-1)
        gate = gate*keep
        gate = gate/gate.sum(-1,keepdim=True).clamp_min(torch.finfo(gate.dtype).tiny)
        return torch.einsum('bs,bsc->bc',gate,logits),gate,keep


def selected_pool(x, keep):
    # keep is straight-through during training; casting it to bool here would
    # again disconnect final survival supervision from selection parameters.
    return (x*keep.unsqueeze(-1)).sum(1)/keep.sum(1,keepdim=True).clamp_min(1.)


class ReadoutFixedSlotSPE(SlotSPE):
    def fuse_slots(self, slots_wsi, slots_omic, keep_wsi, keep_omic):
        cross = self.cross_attention(slots_wsi,slots_omic)
        intra_wsi = self.self_attention_wsi(slots_wsi,mask=keep_wsi.detach()>0.5)
        intra_omic = self.self_attention_omic(slots_omic,mask=keep_omic.detach()>0.5)
        # Keep the original dense cross branch. Only the sparse intra branches
        # acquire an actually selected readout and a surrogate selector gradient.
        return torch.cat([cross.mean(1),selected_pool(intra_wsi,keep_wsi),
                          selected_pool(intra_omic,keep_omic)],dim=1)

    def forward(self, **kwargs):
        x_wsi_proj = self.wsi_mlp(kwargs['x_wsi'])
        if self.args.rna_format != 'Pathways':
            raise ValueError('This scoped proof control supports Pathways only.')
        x_omics = torch.stack([self.sig_networks[i](kwargs[f'x_omic{i+1}'])
                              for i in range(self.num_pathways)],dim=1)
        if not self.training and kwargs.get('omic_missing',False):
            x_omics = self.reconstruction_head_joint(x_omics,self.slot_attention_omic(x_wsi_proj))
        slots_wsi = self.slot_attention_wsi(x_wsi_proj)
        slots_omic = self.slot_attention_omic(x_omics)
        logits_wsi,_,keep_wsi = self.slot_decoder_wsi(slots_wsi)
        logits_omic,_,keep_omic = self.slot_decoder_omic(slots_omic)
        fused = self.fuse_slots(slots_wsi,slots_omic,keep_wsi,keep_omic)
        prediction = self.to_logits(fused)
        if not self.training:
            return prediction,0.0
        target = x_omics.detach()
        recon_omic = self.reconstruction_head_omic(x_omics,slots_omic,mask=keep_omic.detach()>0.5)
        recon_mse = F.mse_loss(recon_omic,target)
        from_wsi = self.slot_attention_omic(x_wsi_proj)
        recon_joint = self.reconstruction_head_joint(x_omics,from_wsi)
        recon_mse = recon_mse+F.mse_loss(recon_joint,target)
        recon_wsi = self.reconstruction_head_wsi(x_wsi_proj,slots_wsi,mask=keep_wsi.detach()>0.5)
        recon_wsi = F.normalize(recon_wsi,dim=-1)
        target_wsi = F.normalize(x_wsi_proj,dim=-1)
        recon_loss = recon_mse+1-F.cosine_similarity(recon_wsi,target_wsi,dim=-1).mean()
        decoder_loss = (self.loss_fn(logits_wsi,y=kwargs['y'],c=kwargs['c'],t=None)+
                        self.loss_fn(logits_omic,y=kwargs['y'],c=kwargs['c'],t=None))/len(kwargs['y'])
        return prediction,decoder_loss+self.args.lambda_recon_loss*recon_loss


def apply_repairs(net, arm):
    if arm == 'original':
        return net
    if arm not in ['eval_fix','readout_fix']:
        raise ValueError(arm)
    if arm == 'readout_fix':
        net.__class__ = ReadoutFixedSlotSPE
    for index,name in enumerate(['slot_attention_wsi','slot_attention_omic']):
        module = getattr(net,name)
        module.__class__ = ReproducibleSlotAttention
        generator = torch.Generator(device='cpu').manual_seed(160018+index)
        noise = torch.randn(1,module.num_slots,module.dim,generator=generator)
        module.register_buffer('_proof_eval_noise',noise,persistent=False)
    for name in ['slot_decoder_wsi','slot_decoder_omic']:
        getattr(net,name).__class__ = StableSlotDecoder
    return net


def build_model(args, arm):
    return apply_repairs(SlotSPE(args),arm)
