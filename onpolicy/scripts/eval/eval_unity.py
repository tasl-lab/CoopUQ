#!/usr/bin/env python
import multiprocessing
import sys
import os
try:
    import wandb          # only needed with --use_wandb
except ImportError:
    wandb = None
import socket
import setproctitle
import numpy as np
from pathlib import Path

import torch

from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from onpolicy.envs.env_continuous import ContinuousActionEnv


from predictor.scripts.model import RVAEP

REPO_ROOT = Path(__file__).resolve().parents[3]


def make_eval_env(all_args):
    env_dir = Path(all_args.unity_env_dir)
    if not env_dir.is_absolute():
        env_dir = REPO_ROOT / env_dir

    custom_env_path = all_args.unity_env_path

    def build_path(rank):
        if all_args.eval_env_family == 'custom':
            # a build made with simulator/build.sh, the same one for every thread
            path = Path(custom_env_path)
            return str(path if path.is_absolute() else REPO_ROOT / path)
        forced = all_args.eval_scene
        scene_id = forced if forced in (1, 2, 3) else (rank % 3) + 1
        build = {'iid': 'iid_{}', 'ood': 'ood_{}', 'ood_faster': 'ood_{}_faster',
                 'ood_denser': 'ood_{}_denser', 'ood_rooms': 'ood_rooms'}[all_args.eval_env_family]
        # every build is named after its number of robots: iid_1_n3, ood_2_faster_n10, ood_rooms_n5, ...
        build = build.format(scene_id) + f'_n{all_args.num_agents}'
        return str(env_dir / (build + '.x86_64'))

    # the build of every thread, saved with --eval_traj_path so that render_2d.py can draw the layout
    all_args.eval_scenes = [build_path(rank) for rank in range(all_args.n_eval_rollout_threads)]

    def get_env_fn(rank):
        def init_env():
            all_args.unity_env_path = all_args.eval_scenes[rank]
            all_args.rank = rank

            env = ContinuousActionEnv(all_args)
            env.seed(all_args.seed + rank * 1000)

            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument('--num_agents', type=int,
                        default=3, help="number of players")
    parser.add_argument('--eval_env_family', type=str, default='iid',
                        choices=['iid', 'ood', 'ood_faster', 'ood_denser', 'ood_rooms', 'custom'],
                        help="evaluation setting: in-distribution, unseen layouts, rushing humans, denser "
                             "crowds, the unseen four-room layout, or a custom build given by --unity_env_path")
    parser.add_argument('--eval_scene', type=int, default=0,
                        help="force a single layout 1/2/3 (0 = thread rank -> layout)")
    parser.add_argument('--eval_hold_steps', type=int, default=0,
                        help="a robot stands still for this many control steps after it is (re)spawned, "
                             "until its observation history belongs to the new episode")
    parser.add_argument('--unity_env_dir', type=str, default='unity_envs',
                        help="directory with the Unity builds (relative to the repo root)")
    parser.add_argument('--pad_humans', type=int, default=15,
                        help="pad the pedestrian block of the observation to this many slots, so that "
                             "layouts with different crowd sizes run side by side (0 = off)")

    parser.add_argument('--eval_traj_path', type=str, default='',
                        help="save the per-step robot and pedestrian positions of the evaluation to this "
                             ".npz, for onpolicy/scripts/render_2d.py")

    all_args = parser.parse_known_args(args)[0]
    all_args.no_graphics = True

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False.")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    assert all_args.use_eval, ("u need to set use_eval be True")
    assert not (all_args.model_dir == None or all_args.model_dir == ""), ("set model_dir first")

    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(
        all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    eval_envs = make_eval_env(all_args)
    envs = eval_envs
    num_agents = all_args.num_agents

    print(all_args)

    checkpoint = torch.load(all_args.prediction_checkpoint_path, map_location=device)
    model = RVAEP(latent_dim=checkpoint['latent_dim']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
        "prediction_model": model,
    }

    assert all_args.share_policy, "only the shared-policy runners are released"
    from onpolicy.runner.shared.unity_runner_cost import UnityCRunner as Runner

    runner = Runner(config)
    runner.eval(all_args.eval_episodes * all_args.episode_length * all_args.n_eval_rollout_threads)

    runner.eval_log_file.close()
    eval_envs.close()
    print("close eval envs")

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main(sys.argv[1:])
    sys.stdout.flush()
    os._exit(0)
