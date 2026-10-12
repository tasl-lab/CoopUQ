import torch
import torch.nn as nn
import torch.nn.functional as F

from predictor.scripts.convlstm import ConvLSTMCell

MAP_SIZE = 64
VIS_STEP = 5
FUTURE_STEP = 5


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_residual_hiddens),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_hiddens)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_hiddens // 2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_hiddens // 2),
            nn.ReLU()
        )
        self._conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=num_hiddens // 2,
                      out_channels=num_hiddens,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_hiddens)
        )
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._conv_2(x)
        x = self._residual_stack(x)
        return x


class Decoder(nn.Module):
    """Decodes one latent map into one OGM, conditioned on the current OGM."""
    def __init__(self, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                 cond_channels):
        super(Decoder, self).__init__()
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        self._conv_trans_2 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=num_hiddens,
                               out_channels=num_hiddens // 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_hiddens // 2),
            nn.ReLU()
        )
        self._upsample_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                               out_channels=num_hiddens // 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_hiddens // 2),
            nn.ReLU(),
        )
        self._final_conv = nn.Sequential(
            nn.Conv2d(in_channels=num_hiddens // 2 + cond_channels,
                      out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, inputs, cond):
        x = self._residual_stack(inputs)
        x = self._conv_trans_2(x)
        x = self._upsample_1(x)
        x = torch.cat([x, cond], dim=1)
        return self._final_conv(x)


class VAE_Encoder(nn.Module):
    def __init__(self, input_channel, embedding_dim):
        super(VAE_Encoder, self).__init__()
        self.input_channels = input_channel
        num_hiddens = 128
        num_residual_hiddens = 64
        num_residual_layers = 2

        self._encoder = Encoder(input_channel,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._encoder_z_mu = nn.Conv2d(in_channels=num_hiddens,
                                       out_channels=embedding_dim,
                                       kernel_size=1, stride=1)
        self._encoder_z_log_sd = nn.Conv2d(in_channels=num_hiddens,
                                           out_channels=embedding_dim,
                                           kernel_size=1, stride=1)

    def forward(self, x):
        x = x.reshape(-1, self.input_channels, MAP_SIZE, MAP_SIZE)
        encoder_out = self._encoder(x)
        z_mu = self._encoder_z_mu(encoder_out)
        z_log_sd = self._encoder_z_log_sd(encoder_out)
        return z_mu, z_log_sd


class _ChannelLN(nn.Module):
    """LayerNorm over the channel dimension of a (B, C, H, W) map."""
    def __init__(self, C):
        super().__init__()
        self.ln = nn.LayerNorm(C)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DeviationGuidedFusion(nn.Module):
    """Deviation-guided cross-robot fusion.

    The spatial disagreement between the aligned robot features is turned into a
    learnable gate that biases the cross-agent spatial attention of the ego robot,
    followed by window-based multi-head self-attention and a feed-forward network:
        y   = ego + alpha * MHSA(CrossAgentAttn(features))
        out = y + FFN(LN(y))
    """
    def __init__(self, num_channels, heads=4, window=8, ffn_mult=2):
        super().__init__()
        C = num_channels
        self.heads = heads
        self.window = window
        self.hd = C // heads
        self.ln_in = _ChannelLN(C)
        self.ca_q = nn.Conv2d(C, C, 1)
        self.ca_k = nn.Conv2d(C, C, 1)
        self.ca_v = nn.Conv2d(C, C, 1)
        self.ca_scale = self.hd ** -0.5
        self.sa_qkv = nn.Conv2d(C, 3 * C, 1)
        self.sa_scale = self.hd ** -0.5
        self.alpha = nn.Parameter(torch.zeros(1))
        self.ln = _ChannelLN(C)
        self.ffn = nn.Sequential(nn.Conv2d(C, C * ffn_mult, 1), nn.GELU(),
                                 nn.Conv2d(C * ffn_mult, C, 1))
        self.g_b = nn.Parameter(torch.tensor(0.0))
        self.g_dis = nn.Parameter(torch.tensor(1.0))

    def _win_mhsa(self, x):
        B, C, H, W = x.shape
        win = self.window
        ph = (win - H % win) % win
        pw = (win - W % win) % win
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        Hh, Ww = H + ph, W + pw
        nh, nw = Hh // win, Ww // win
        qkv = self.sa_qkv(x).reshape(B, 3 * C, nh, win, nw, win)
        qkv = qkv.permute(0, 2, 4, 3, 5, 1).reshape(B * nh * nw, win * win, 3 * C)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t):
            return t.reshape(t.shape[0], win * win, self.heads, self.hd).permute(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        a = F.softmax((q @ k.transpose(-1, -2)) * self.sa_scale, dim=-1)
        o = (a @ v).permute(0, 2, 1, 3).reshape(B * nh * nw, win * win, C)
        o = o.reshape(B, nh, nw, win, win, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, Hh, Ww)
        return o[..., :H, :W] if (ph or pw) else o

    def forward(self, features):
        """features: (N, B, C, H, W), aligned to the ego frame, ego first."""
        N, B, C, H, W = features.shape
        ego = features[0]
        heads, hd = self.heads, self.hd
        fn = self.ln_in(features.reshape(N * B, C, H, W)).reshape(N, B, C, H, W)
        qh = self.ca_q(fn[0]).reshape(B, heads, hd, H, W)
        flat = fn.reshape(N * B, C, H, W)
        kh = self.ca_k(flat).reshape(N, B, heads, hd, H, W)
        vh = self.ca_v(flat).reshape(N, B, heads, hd, H, W)
        scores = (qh.unsqueeze(0) * kh).sum(dim=3, keepdim=True) * self.ca_scale

        mean = features.mean(0)
        dis = ((features - mean.unsqueeze(0)) ** 2).mean(0).mean(1, keepdim=True)
        dis = dis / (dis.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        bias = torch.zeros_like(scores)
        bias[0] = (self.g_b - self.g_dis * dis).unsqueeze(1)
        scores = scores + bias

        ca = (F.softmax(scores, dim=0) * vh).sum(dim=0).reshape(B, C, H, W)
        sa = self._win_mhsa(ca)
        y = ego + self.alpha * sa
        return y + self.ffn(self.ln(y))


def warp_features_translate(feature, tx_pix, ty_pix):
    """Shift a (B, C, H, W) map by (tx_pix, ty_pix) pixels.

    tx_pix is along the H (row) axis and ty_pix along the W (column) axis: the
    content at (r, c) moves to (r + tx_pix, c + ty_pix).
    """
    B, C, H, W = feature.shape
    device, dtype = feature.device, feature.dtype
    tx_pix = tx_pix.to(device=device, dtype=dtype)
    ty_pix = ty_pix.to(device=device, dtype=dtype)
    theta = torch.zeros(B, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = -2.0 * ty_pix / max(W - 1, 1)
    theta[:, 1, 2] = -2.0 * tx_pix / max(H - 1, 1)
    grid = F.affine_grid(theta, feature.shape, align_corners=True)
    return F.grid_sample(feature, grid, mode='bilinear',
                         padding_mode='zeros', align_corners=True)


class LearnableAlignResidual(nn.Module):
    """Predicts a small pixel offset that refines the pose-based feature alignment."""
    def __init__(self, feat_ch, hidden=64, max_shift=2.0):
        super().__init__()
        self.max_shift = float(max_shift)
        self.net = nn.Sequential(
            nn.Conv2d(2 * feat_ch, hidden, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(hidden, 2)

    def forward(self, ego_feat, warped_peer):
        z = self.net(torch.cat([ego_feat, warped_peer], dim=1)).flatten(1)
        d = torch.tanh(self.head(z)) * self.max_shift
        return d[:, 0], d[:, 1]


class RVAEP(nn.Module):
    """Cooperative OGM forecaster.

    Pipeline for one ego robot:
        encode_temporal   per-robot ConvLSTM encoding of the OGM history
        align_feature     pose-based warp of a neighbour feature + learnable residual
        fuse              deviation-guided cross-robot fusion
        decode_from_fused VAE with one latent per future step, decoded into
                          `output_channels` future OGMs conditioned on the current OGM
    """
    def __init__(self, input_channels=1, latent_dim=2048, output_channels=FUTURE_STEP):
        super(RVAEP, self).__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.output_channels = output_channels

        num_hiddens = 128
        num_residual_hiddens = 64
        num_residual_layers = 2
        feat_ch = num_hiddens // 4

        self.z_w = MAP_SIZE // 4
        assert latent_dim % (self.z_w * self.z_w) == 0, \
            f"latent_dim {latent_dim} must be a multiple of {self.z_w * self.z_w}"
        self.embedding_dim = latent_dim // (self.z_w * self.z_w)

        self._convlstm = ConvLSTMCell(input_dim=input_channels,
                                      hidden_dim=feat_ch,
                                      kernel_size=(3, 3),
                                      bias=True)
        self.align_residual = LearnableAlignResidual(feat_ch)
        self._fusion = DeviationGuidedFusion(num_channels=feat_ch)
        self._encoder = VAE_Encoder(feat_ch + input_channels,
                                    embedding_dim=self.embedding_dim * output_channels)
        self._time_embed = nn.Parameter(
            torch.randn(output_channels, self.embedding_dim, 1, 1) * 0.02)
        self._decoder_z_mu = nn.ConvTranspose2d(in_channels=self.embedding_dim,
                                                out_channels=num_hiddens,
                                                kernel_size=1, stride=1)
        self._decoder = Decoder(1,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens,
                                cond_channels=input_channels)

    def vae_reparameterize(self, z_mu, z_log_sd):
        z_mu = z_mu.reshape(-1, self.latent_dim, 1)
        z_log_sd = z_log_sd.reshape(-1, self.latent_dim, 1)
        pz = torch.distributions.Normal(loc=torch.zeros_like(z_mu),
                                        scale=torch.ones_like(z_log_sd))
        qz_x = torch.distributions.Normal(loc=z_mu, scale=torch.exp(z_log_sd))
        z = qz_x.rsample()
        kl_loss = (qz_x.log_prob(z) - pz.log_prob(z)).sum(dim=1).mean()
        return z, kl_loss

    def encode_temporal(self, x):
        """x: (B, VIS_STEP, 1, MAP_SIZE, MAP_SIZE) -> feature (B, 32, MAP_SIZE, MAP_SIZE)."""
        x = x.reshape(-1, VIS_STEP, 1, MAP_SIZE, MAP_SIZE)
        b, seq_len, c, h, w = x.size()
        h_enc, enc_state = self._convlstm.init_hidden(batch_size=b, image_size=(h, w))
        for t in range(seq_len):
            h_enc, enc_state = self._convlstm(input_tensor=x[:, t],
                                              cur_state=[h_enc, enc_state])
        return h_enc

    def align_feature(self, peer_feat, ego_feat, tx_pix, ty_pix):
        """Warp a neighbour feature into the ego frame given the relative pose in pixels."""
        warped = warp_features_translate(peer_feat, tx_pix, ty_pix)
        dtx, dty = self.align_residual(ego_feat, warped)
        return warp_features_translate(peer_feat, tx_pix + dtx, ty_pix + dty)

    def fuse(self, features):
        """features: (N, B, C, H, W), aligned to the ego frame, ego first."""
        return self._fusion(features)

    def decode_from_fused(self, h_fused, x_map):
        """Sample one forecast.

        h_fused: (B, 32, MAP_SIZE, MAP_SIZE) fused feature.
        x_map:   (B, 1, MAP_SIZE, MAP_SIZE) current OGM.
        Returns the forecast (B, output_channels, MAP_SIZE, MAP_SIZE) and the KL term.
        """
        x_map = x_map.reshape(-1, self.input_channels, MAP_SIZE, MAP_SIZE)
        z_mu, z_log_sd = self._encoder(torch.cat([h_fused, x_map], dim=1))
        z, kl_loss = self.vae_reparameterize(z_mu, z_log_sd)

        B = h_fused.shape[0]
        T = self.output_channels
        ed = self.embedding_dim
        z = z.reshape(B * T, ed, self.z_w, self.z_w)
        z = z + self._time_embed.unsqueeze(0).expand(B, T, ed, 1, 1).reshape(B * T, ed, 1, 1)
        x_d = self._decoder_z_mu(z)
        cond = x_map.unsqueeze(1).expand(B, T, *x_map.shape[1:]).reshape(B * T, *x_map.shape[1:])
        prediction = self._decoder(x_d, cond=cond)
        return prediction.reshape(B, T, MAP_SIZE, MAP_SIZE), kl_loss
