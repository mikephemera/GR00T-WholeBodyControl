import os
import numpy as np
import mujoco
from types import SimpleNamespace
import torch as t
from motionbricks.motion_backbone.inference.motion_inference import motion_inference
from motionbricks.motion_backbone.demo.controllers import WASD_controller, random_controller
from motionbricks.exp_setup.experiment import test
import xml.etree.ElementTree as ET
from copy import deepcopy

class navigation_demo(object):
    def __init__(self, args):
        self.args = args
        self.full_agent = None
        self.controller = None
        self.mj_model = None
        self.mj_data = None
        self._parse_args()
        self._initialize_inference_modles()
        self._initialize_controller()
        self._initialize_mj_simulator()

    def _parse_args(self):
        self.args.return_model_configs = True
        self.args.return_dataloader = True

        # parse the default path if not given (very likely used by an external project)
        # Navigate from motionbricks/motion_backbone/demo/utils.py up to the project root
        project_base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        if not hasattr(self.args, 'humanoid_scene_xml'):
            self.args.humanoid_scene_xml = \
                os.path.abspath(os.path.join(project_base_path, "assets", "skeletons", "g1", "scene_29dof.xml"))

        if not hasattr(self.args, 'skeleton_xml'):
            self.args.skeleton_xml = \
                os.path.abspath(os.path.join(project_base_path, "assets", "skeletons", "g1", "g1.xml"))

        if not hasattr(self.args, 'result_dir'):
            self.args.result_dir = os.path.abspath(os.path.join(project_base_path, "out"))

        if not hasattr(self.args, 'data_root'):
            self.args.data_root = os.path.abspath(os.path.join(project_base_path, "datasets"))

        if not hasattr(self.args, 'clips_ckpt'):
            result_dir = getattr(self.args, 'result_dir', os.path.join(project_base_path, "out"))
            self.args.clips_ckpt = os.path.abspath(os.path.join(result_dir, "G1-clip.ckpt"))

        if not hasattr(self.args, 'explicit_dataset_folder'):
            self.args.explicit_dataset_folder = \
                os.path.abspath(os.path.join(project_base_path, "datasets", "motionbricks-G1"))

    def _initialize_inference_modles(self):
        reprocess_clips = getattr(self.args, 'reprocess_clips', False)  # useful for debugging & development
        if self.args.clips_ckpt is None or (not os.path.exists(self.args.clips_ckpt)) or reprocess_clips:
            models, confs, train_dataloader, val_dataloader = test(self.args)
            self.args.train_dataloader = train_dataloader
            self.args.val_dataloader = val_dataloader
        else:
            self.args.return_dataloader = False
            models, confs = test(self.args)
            self.args.train_dataloader = None
            self.args.val_dataloader = None

        for model_name in ['pose', 'root']:
            state_dict = t.load(confs[model_name].ckpt_path)['state_dict']
            models[model_name].load_state_dict(state_dict)
        self.inferencer = motion_inference(models, models['pose'].args)

        from motionbricks.motion_backbone.demo.full_agent import full_navigation_agent
        target_root_realignment = getattr(self.args, 'target_root_realignment', True)
        source_root_realignment = getattr(self.args, 'source_root_realignment', True)
        force_canonicalization = getattr(self.args, 'force_canonicalization', True)
        skip_ending_target_cond = getattr(self.args, 'skip_ending_target_cond', False)
        speed_scale = getattr(self.args, 'speed_scale', [0.8, 1.2]) if \
            getattr(self.args, 'random_speed_scale', False) else [1.0, 1.0]
        self.full_agent = full_navigation_agent(self.inferencer, self.args.train_dataloader, device='cuda',
                                                speed_scale=speed_scale,
                                                target_root_realignment=target_root_realignment,
                                                source_root_realignment=source_root_realignment,
                                                force_canonicalization=force_canonicalization,
                                                skeleton_xml=self.args.skeleton_xml,
                                                skip_ending_target_cond=skip_ending_target_cond,
                                                filter_qpos=getattr(self.args, 'pre_filter_qpos', True),
                                                clips=self.args.clips,
                                                ckpt_path=self.args.clips_ckpt,
                                                reprocess_clips=reprocess_clips,
                                                val_dataloader=self.args.val_dataloader).to('cuda')

    def _initialize_controller(self):
        lookat_movement_direction = getattr(self.args, 'lookat_movement_direction', False)
        min_tokens = self.inferencer._args['min_tokens']
        max_tokens = self.inferencer._args['max_tokens']

        if self.args.controller == "wasd":
            self.controller = WASD_controller(lookat_movement_direction=lookat_movement_direction,
                                              clips=self.args.clips, min_token=min_tokens, max_token=max_tokens)

        elif self.args.controller == "random":
            max_angle_change_between_controls = getattr(self.args, 'max_angle_change_between_controls', 0.5 * np.pi)
            self.controller = random_controller(disable_running=getattr(self.args, 'disable_running', True),
                                                lookat_movement_direction=lookat_movement_direction,
                                                new_control_dt=getattr(self.args, 'new_control_dt', 2.0),
                                                max_angle_change_between_controls=max_angle_change_between_controls,
                                                clips=self.args.clips, min_token=min_tokens, max_token=max_tokens)

        else:
            raise ValueError(f"Controller {self.args.controller} is not supported")

    def _initialize_mj_simulator(self):
        add_ghosts = getattr(self.args, 'dryrun', False)
        result = build_mj_simulator(self.args.humanoid_scene_xml, self.inferencer.motion_rep.fps,
                                    add_ghosts=add_ghosts)
        if add_ghosts:
            self.mj_model, self.mj_data, self._ghost_qpos_ranges = result
        else:
            self.mj_model, self.mj_data = result
            self._ghost_qpos_ranges = None


def _prepend_names(elem, prefix):
    """Recursively prepend prefix to all 'name' attributes in the element tree."""
    if 'name' in elem.attrib:
        elem.attrib['name'] = prefix + elem.attrib['name']
    for child in elem:
        _prepend_names(child, prefix)


def _replace_attribute(elem, attribute, value):
    """Recursively replace an attribute value in the element tree."""
    if attribute in elem.attrib:
        elem.attrib[attribute] = value
    for child in elem:
        _replace_attribute(child, attribute, value)


def build_ghost_augmented_scene_xml(scene_xml_path: str) -> tuple:
    """Build a MuJoCo scene XML string that includes the original robot plus two
    semi-transparent ghost robots (red=context, green=target).

    Returns:
        (xml_string, ghost_joint_names_red, ghost_joint_names_green)
        where ghost_joint_names_* are lists of (joint_name, qpos_index) for each ghost.
    """
    # Parse scene and robot XMLs (use absolute paths for from_xml_string compatibility)
    scene_dir = os.path.abspath(os.path.dirname(scene_xml_path))
    robot_xml_path = os.path.join(scene_dir, "g1_29dof.xml")

    scene_tree = ET.parse(scene_xml_path)
    scene_root = scene_tree.getroot()

    robot_tree = ET.parse(robot_xml_path)
    robot_root = robot_tree.getroot()

    # Merge <compiler> settings from robot to scene (needed when <include> is removed).
    # CRITICAL: remove meshdir because we set absolute mesh paths below.
    robot_compiler = robot_root.find('compiler')
    if robot_compiler is not None:
        scene_compiler = scene_root.find('compiler')
        if scene_compiler is None:
            scene_compiler = ET.Element('compiler')
            scene_root.insert(0, scene_compiler)
        for attr, val in robot_compiler.attrib.items():
            if attr != 'meshdir':
                scene_compiler.set(attr, val)

    # Merge <asset> meshes: copy mesh elements from robot to scene
    robot_asset = robot_root.find('asset')
    if robot_asset is not None:
        scene_asset = scene_root.find('asset')
        if scene_asset is None:
            scene_asset = ET.SubElement(scene_root, 'asset')
        mesh_dir = os.path.join(scene_dir, 'meshes')
        for mesh in robot_asset.findall('mesh'):
            mesh_copy = deepcopy(mesh)
            # Update mesh file path to be relative to the scene XML
            mesh_file = mesh_copy.get('file', '')
            if not os.path.isabs(mesh_file):
                mesh_copy.set('file', os.path.join(mesh_dir, mesh_file))
            scene_asset.append(mesh_copy)

    # Merge <default> classes from robot to scene
    robot_default = robot_root.find('default')
    if robot_default is not None:
        scene_default = scene_root.find('default')
        if scene_default is None:
            scene_default = ET.SubElement(scene_root, 'default')
        for default_elem in robot_default.findall('default'):
            scene_default.append(deepcopy(default_elem))

    # Get the worldbody
    scene_worldbody = scene_root.find('worldbody')
    if scene_worldbody is None:
        scene_worldbody = ET.SubElement(scene_root, 'worldbody')

    # Extract original robot body tree
    robot_worldbody = robot_root.find('worldbody')
    original_pelvis = robot_worldbody.find('body')  # The pelvis body
    if original_pelvis is None:
        raise ValueError("No <body> found in robot XML worldbody")

    # Remove ALL <include> elements from the scene root (they are top-level children of <mujoco>)
    for include_elem in list(scene_root.findall('include')):
        scene_root.remove(include_elem)
    # Also remove any existing body named "pelvis" from the scene worldbody (from include expansion)
    for body_elem in list(scene_worldbody.findall('body')):
        if body_elem.get('name') == 'pelvis':
            scene_worldbody.remove(body_elem)

    scene_worldbody.append(deepcopy(original_pelvis))

    # Create red ghost (context) — rgba red, semi-transparent
    red_pelvis = deepcopy(original_pelvis)
    _prepend_names(red_pelvis, "ghost_red_")
    _replace_attribute(red_pelvis, "rgba", "0.85 0.10 0.10 0.35")
    scene_worldbody.append(red_pelvis)

    # Create green ghost (target) — rgba green, semi-transparent
    green_pelvis = deepcopy(original_pelvis)
    _prepend_names(green_pelvis, "ghost_green_")
    _replace_attribute(green_pelvis, "rgba", "0.10 0.85 0.10 0.35")
    scene_worldbody.append(green_pelvis)

    # Also copy <actuator> and <sensor> from robot (only for the main robot, not ghosts)
    for section_name in ['actuator', 'sensor']:
        robot_section = robot_root.find(section_name)
        if robot_section is not None:
            scene_section = scene_root.find(section_name)
            if scene_section is not None:
                scene_root.remove(scene_section)
            scene_root.append(deepcopy(robot_section))

    xml_string = ET.tostring(scene_root, encoding='unicode')
    return xml_string


def build_mj_simulator(humanoid_xml: str, fps: int = 30, build_dummy_mj_simulator: bool = False,
                       add_ghosts: bool = False):
    if build_dummy_mj_simulator:
        mj_model = SimpleNamespace(opt=SimpleNamespace(timestep=1 / fps))
        mj_data = SimpleNamespace(qpos=np.zeros(36))
        if add_ghosts:
            return mj_model, mj_data, {}
        return mj_model, mj_data

    if add_ghosts:
        xml_string = build_ghost_augmented_scene_xml(humanoid_xml)
        mj_model = mujoco.MjModel.from_xml_string(xml_string)
    else:
        mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)

    mj_data = mujoco.MjData(mj_model)
    # Disable advanced visual effects for better performance
    mj_model.vis.global_.offwidth = 1920
    mj_model.vis.global_.offheight = 1080
    mj_model.vis.quality.shadowsize = 0

    mj_model.vis.rgba.fog = [0, 0, 0, 0]

    mj_model.vis.headlight.ambient = [0.8, 0.8, 0.8]
    mj_model.vis.headlight.diffuse = [0.8, 0.8, 0.8]
    mj_model.vis.headlight.specular = [0.1, 0.1, 0.1]

    mj_model.opt.timestep = 1 / fps

    if add_ghosts:
        # Find qpos address ranges for each ghost by joint name
        ghost_qpos_ranges = {'red': {}, 'green': {}}
        for prefix, key in [('ghost_red_', 'red'), ('ghost_green_', 'green')]:
            # Find the free joint
            free_joint_name = prefix + 'floating_base_joint'
            try:
                free_joint_id = mj_model.joint(free_joint_name).id
                free_joint_qpos = mj_model.jnt_qposadr[free_joint_id]
                ghost_qpos_ranges[key]['free_start'] = free_joint_qpos
                ghost_qpos_ranges[key]['free_dof'] = 7
            except KeyError:
                pass

            # Find hinge joints (29 DOF)
            hinge_start = None
            hinge_count = 0
            for j in range(mj_model.njnt):
                jname = mj_model.joint(j).name
                if jname.startswith(prefix) and mj_model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
                    qadr = mj_model.jnt_qposadr[j]
                    if hinge_start is None:
                        hinge_start = qadr
                    hinge_count += 1
            ghost_qpos_ranges[key]['hinge_start'] = hinge_start
            ghost_qpos_ranges[key]['hinge_count'] = hinge_count

        return mj_model, mj_data, ghost_qpos_ranges

    return mj_model, mj_data


