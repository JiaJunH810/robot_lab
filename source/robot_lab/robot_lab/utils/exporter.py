# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
import copy
import onnx

from isaaclab.envs import ManagerBasedRLEnv
from robot_lab.utils.vae_exporter import VaeJitPolicyExporter, VaeOnnxPolicyExporter

def export_vae_policy_as_jit(policy: object, normalizer: object | None, path: str, filename="policy.pt"):
    exporter = VaeJitPolicyExporter(policy, normalizer)
    exporter.export(path, filename)

def export_vae_policy_as_onnx(policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = VaeOnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )

def attach_onnx_metadata(env: ManagerBasedRLEnv, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)
    metadata = {
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": env.observation_manager.active_terms["policy"],
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        "body_names": env.scene["robot"].data.body_names,
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)