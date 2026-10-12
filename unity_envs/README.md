# Simulation environments

Download `CoopUQ_unity_envs.tar.gz` from [Hugging Face](https://huggingface.co/datasets/tasl-lab/CoopUQ-unity-envs)
(see the main README) and unpack it in the repository root; it extracts into this directory:

```bash
tar -xzf CoopUQ_unity_envs.tar.gz
chmod +x unity_envs/*.x86_64
```

There is one build per layout (`k = 1, 2, 3`) of each evaluation setting:

| Setting (`--eval_env_family`) | Builds | Description |
|---|---|---|
| `iid` | `iid_k_n3.x86_64` | training layouts, 7-10 pedestrians |
| `ood` | `ood_k_n3.x86_64` | unseen layouts |
| `ood_faster` | `ood_k_faster_n3.x86_64` | rushing pedestrians, up to 2 m/s |
| `ood_denser` | `ood_k_denser_n3.x86_64` | denser crowds, 12-15 pedestrians |

Each build `NAME.x86_64` comes with its data folder `NAME_Data/`; all builds share `UnityPlayer.so`.
