#This file is partly constructed based on 
#MST++: https://github.com/caiyuanhao1998/MST-plus-plus/blob/master/predict_code/architecture/MST_Plus_Plus.py
#and Uformer: https://github.com/ZhendongWang6/Uformer/blob/main/model.py

import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
from timm.models.layers import  to_2tuple

class IPT_dehazenet(nn.Module):
    
    def __init__(self, in_channels=172, out_channels=172, n_feat=172):
        
        super(IPT_dehazenet, self).__init__()
        
        self.autobandselection=autobandselection()
        self.spectral_rec=spectral_rec()
        self.conv_in = nn.Conv2d(in_channels*2, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)      
        self.spectral_refinement1 = SMSAB()
        self.spatial_refinement1 = WMSAB()
        self.spectral_refinement2 = SMSAB()
        self.spatial_refinement2 = WMSAB()      
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)
      
    def forward(self, xin):

        y=self.autobandselection(xin)#ABS
        xou=self.spectral_rec(y)#SR     
        x_resi=torch.cat((xin,xou),1)        
        x_emb = self.conv_in(x_resi)#embedding 
        h = self.spectral_refinement1(x_emb)
        h = self.spatial_refinement1(h)
        h = self.spectral_refinement2(h)
        h = self.spatial_refinement2(h)      
        h = self.conv_out(h)
        h += x_emb
         
        return h,y

class autobandselection(nn.Module):
    
    def __init__(self,):
        super(autobandselection, self).__init__()
        
        self.depthconv = nn.Conv2d(in_channels=172,out_channels=172,kernel_size=1,groups=172)                                         
        self.relu = nn.Sequential( nn.ReLU(),)
                
    def forward(self, x):

        x = self.depthconv(x)
        x=self.relu(x)#non-negative constraint
        
        return  x
        
class GELU(nn.Module):
    
    def forward(self, x):
        
        return F.gelu(x)
      
class spectral_rec(nn.Module):
    
    def __init__(self, ):
        super(spectral_rec, self).__init__()
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=172,out_channels=172,kernel_size=3,padding=1),
                                          GELU(),)

        self.lr_module = nn.Sequential(nn.Conv2d(in_channels=172,out_channels=10,kernel_size=1),
                                          GELU(),)
        
        self.spectral_rec=nn.Sequential(nn.Conv2d(in_channels=10,out_channels=10,kernel_size=1,groups=10),
                                          GELU(),
                                          nn.Conv2d(in_channels=10,out_channels=172,kernel_size=3,padding=1),
                                          GELU(),
                                          nn.Conv2d(in_channels=172,out_channels=172,kernel_size=1),
                                          GELU(),)
        
    def forward(self, x):
        
        x = self.conv3(x)
        x = self.lr_module(x)
        x = self.spectral_rec(x) 
        
        return x
    
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super(FeedForward,self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)
    
class PreNorm(nn.Module):
    
    def __init__(self, dim, fn):
        super(PreNorm,self).__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super(MS_MSA,self).__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p
 
        return out

class SMSAB(nn.Module):
    def __init__(
            self,
            dim=172,
            dim_head=172,
            heads=1,
            num_blocks=1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim,MS_MSA(dim=dim, dim_head=dim_head, heads=heads)),
                PreNorm(dim,FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        
        x = x.permute(0, 2, 3, 1)
        
        for (attn, ff) in self.blocks:

            x = attn(x) + x
            x = ff(x) + x

        out = x.permute(0, 3, 1, 2)

        return out
    
class WMSAB(nn.Module):
    def __init__(
            self,
            dim=172,
            dim_head=172,
            heads=1,
            num_blocks=1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.layerNorm=nn.LayerNorm(dim)
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([

                W_MSA(dim=dim, dim_head=dim_head, heads=heads),#Prenorm defined inside
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """

        B,C,H,W = x.shape
        
        for (attn, ff) in self.blocks:

            y=attn(x)
            y=y.reshape(B,H,W,C)
            x = y + x.permute(0, 2, 3, 1)
            x = ff(x) + x
            out = x.permute(0, 3, 1, 2)
            
        return out

class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super(LinearProjection,self).__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1] 
        return q,k,v

class WindowAttention(nn.Module):
    def __init__(self, dim, win_size,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super(WindowAttention,self).__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.53
        self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.rescale = nn.Parameter(torch.ones(1, 1, 1))

    def forward(self, x, attn_kv=None, mask=None):
        
        B_, N, C = x.shape
        q, k, v = self.qkv(x,attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn=attn*self.rescale
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        return x

def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class LeWinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,token_projection='linear',token_mlp='leff',
                 ):
        super(LeWinTransformerBlock,self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection)

    def forward(self, x, mask=None):
   
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))      
        x = x.view(B, H, W, C)
        x = self.norm1(x)#PreNorm
        shifted_x = x
        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
        wmsa_in = x_windows
        # W-MSA/SW-MSA
        attn_windows = self.attn(wmsa_in, mask=None)  # nW*B, win_size*win_size, C
        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C
        x = shifted_x

        return x

class BasicUformerLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear',token_mlp='ffn', shift_flag=True,
                 ):

        super(BasicUformerLayer,self).__init__()
        self.dim = dim 
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        if shift_flag:
            self.blocks = nn.ModuleList([
                LeWinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, win_size=win_size,
                                    shift_size=0,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=None,
                                    norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,
                                    )]) 

    def forward(self, x, mask=None):
        
        for blk in self.blocks:
            x = blk(x,mask)

        return x

class W_MSA(nn.Module):
    def __init__(self, img_size=256, in_chans=172, dd_in=172,
                 embed_dim=172, depths=1, num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='leff',
                  shift_flag=True, **kwargs):
        super(W_MSA,self).__init__()
        
        # Encoder
        self.encoderlayer_0 = BasicUformerLayer(dim=embed_dim,
                            output_dim=embed_dim,
                            input_resolution=(img_size,
                                                img_size),
                            depth=None,
                            num_heads=num_heads[0],
                            win_size=win_size,
                            mlp_ratio=None,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=None,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag)

    def forward(self, x, mask=None):

        x = x.flatten(2).transpose(1, 2).contiguous()
        conv0 = self.encoderlayer_0(x,mask=mask)

        return conv0