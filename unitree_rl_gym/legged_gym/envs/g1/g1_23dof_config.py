from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class G1_23dofCfg( LeggedRobotCfg ):
    """G1 23-DoF (g1_23dof_rev_1_0) velocity-command walking, contract-v0 aligned.

    Differences vs the 12-DoF G1RoughCfg (all tagged [g1-23dof]):
      - asset g1_23dof_rev_1_0.urdf (23 actuated joints, arms + waist active)
      - 78-dim observation in contract-v0 order, no gait clock (see g1_23dof_env.py)
      - feedforward MLP policy (ActorCritic, no LSTM) -> runner cfg below
      - uniform PD gains Kp=60 / Kd=1.5 for all joints (contract v0)
      - gait rewards: phase-locked `contact` disabled (clock not observed);
        free-form gait via standard `feet_air_time` reward instead
    """

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           # legs (same as 12-DoF variant)
           'left_hip_pitch_joint' : -0.1,
           'left_hip_roll_joint' : 0,
           'left_hip_yaw_joint' : 0.,
           'left_knee_joint' : 0.3,
           'left_ankle_pitch_joint' : -0.2,
           'left_ankle_roll_joint' : 0,
           'right_hip_pitch_joint' : -0.1,
           'right_hip_roll_joint' : 0,
           'right_hip_yaw_joint' : 0.,
           'right_knee_joint' : 0.3,
           'right_ankle_pitch_joint' : -0.2,
           'right_ankle_roll_joint' : 0,
           # waist + arms (zeros = straight, matches URDF zero pose)
           'waist_yaw_joint' : 0.,
           'left_shoulder_pitch_joint' : 0.,
           'left_shoulder_roll_joint' : 0.,
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint' : 0.,
           'left_wrist_roll_joint' : 0.,
           'right_shoulder_pitch_joint' : 0.,
           'right_shoulder_roll_joint' : 0.,
           'right_shoulder_yaw_joint' : 0.,
           'right_elbow_joint' : 0.,
           'right_wrist_roll_joint' : 0.,
        }

    class env(LeggedRobotCfg.env):
        num_observations = 78   # [cmd(3) | ang_vel(3) | gravity(3) | dof_pos(23) | dof_vel(23) | prev_action(23)]
        num_privileged_obs = None  # symmetric actor/critic observation
        num_actions = 23

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
        # [g1-baseline] additions over upstream defaults (kept from the 12-DoF run):
        randomize_gains = True          # per-env multiplicative PD gain randomization
        gain_multiplier_range = [0.9, 1.1]
        randomize_action_delay = True   # per-env action delay in policy steps (20 ms each)
        action_delay_steps_range = [0, 2]

    class control( LeggedRobotCfg.control ):
        control_type = 'P'
        # [g1-23dof] uniform gains per contract v0 (all 23 joints)
        stiffness = {'hip_pitch': 60,
                     'hip_roll': 60,
                     'hip_yaw': 60,
                     'knee': 60,
                     'ankle_pitch': 60,
                     'ankle_roll': 60,
                     'waist_yaw': 60,
                     'shoulder_pitch': 60,
                     'shoulder_roll': 60,
                     'shoulder_yaw': 60,
                     'elbow': 60,
                     'wrist_roll': 60,
                     }  # [N*m/rad]
        damping = {  'hip_pitch': 1.5,
                     'hip_roll': 1.5,
                     'hip_yaw': 1.5,
                     'knee': 1.5,
                     'ankle_pitch': 1.5,
                     'ankle_roll': 1.5,
                     'waist_yaw': 1.5,
                     'shoulder_pitch': 1.5,
                     'shoulder_roll': 1.5,
                     'shoulder_yaw': 1.5,
                     'elbow': 1.5,
                     'wrist_roll': 1.5,
                     }  # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_23dof_rev_1_0.urdf'
        name = "g1_23dof"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis", "torso"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78

        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 1.0     # [g1-23dof] free-form gait reward (no phase clock in obs)
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.0           # [g1-23dof] phase-locked gait reward disabled (clock unobserved)


class G1_23dofCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [128, 64]
        critic_hidden_dims = [128, 64]
        activation = 'elu'

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCritic"   # [g1-23dof] feedforward MLP (contract v0)
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1_23dof'
